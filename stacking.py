"""
stacking.py
Responsible for:
  1. Generate Out-of-Fold (OOF) predictions from LSTM + ARIMA
  2. Train Meta Model (Logistic Regression) on OOF predictions
  3. Final stacking prediction on test set
  4. Save / Load Meta Model

Updated to work with signal-day training logic.
"""

import os, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from lstm_model  import build_model, make_sequences
from arima_model import predict_one


# ════════════════════════════════════════════════════════
#  OOF generation (signal-day aware)
# ════════════════════════════════════════════════════════
def generate_oof(full_df,
                 signal_df,
                 feature_cols: list,
                 split_idx: int,
                 seq_len: int,
                 n_folds: int,
                 arima_order: tuple,
                 epochs: int,
                 batch_size: int,
                 hold_days: int = 5):
    """
    Generate Out-of-Fold predictions on TRAINING SET only.

    Key difference from v1:
      - full_df   : all trading days (for feature sequences)
      - signal_df : only Signal_buy=1 days (for labels)
      - KFold is applied to signal days only
      - Each fold trains LSTM on signal days, predicts held-out signal days
    """
    print(f"\n{'='*55}")
    print(f"  [Stacking] Generating OOF ({n_folds}-Fold, signal days)")
    print(f"{'='*55}")

    # Training signal days only
    signal_train = signal_df[signal_df.index < full_df.index[split_idx]].copy()
    n_signals    = len(signal_train)

    if n_signals < n_folds * 2:
        print(f"  WARNING: Only {n_signals} signal samples, skipping OOF")
        return np.array([]), np.array([]), np.array([])

    # Full feature matrix for sequence context
    X_full  = full_df[feature_cols].values
    close_s = full_df["Close"].squeeze().values
    y_all   = signal_train["Label"].values

    # Get positions of signal days in full_df
    signal_positions = np.array([
        full_df.index.get_loc(idx) for idx in signal_train.index
    ])

    # Placeholders
    oof_lstm_prob  = np.zeros(n_signals)
    oof_arima_pred = np.zeros(n_signals)
    oof_mask       = np.zeros(n_signals, dtype=bool)

    kf = KFold(n_splits=n_folds, shuffle=False)

    for fold, (tr_fold_idx, val_fold_idx) in enumerate(kf.split(np.arange(n_signals))):
        print(f"\n  -- Fold {fold+1}/{n_folds} --")
        print(f"  Signal Train: {len(tr_fold_idx)}  Signal Val: {len(val_fold_idx)}")

        # Signal positions for this fold
        tr_sig_pos  = signal_positions[tr_fold_idx]   # positions in full_df
        val_sig_pos = signal_positions[val_fold_idx]

        y_tr_fold = y_all[tr_fold_idx]
        y_va_fold = y_all[val_fold_idx]

        # Normalize using full_df training portion
        scaler_fold = MinMaxScaler()
        X_full_s    = np.zeros_like(X_full, dtype=float)
        X_full_s[:split_idx]  = scaler_fold.fit_transform(X_full[:split_idx])
        X_full_s[split_idx:]  = scaler_fold.transform(X_full[split_idx:])

        # Build sequences for signal days
        X_tr_seq, y_tr_seq = make_sequences(X_full_s, tr_sig_pos,  y_tr_fold, seq_len, len(X_full))
        X_va_seq, y_va_seq = make_sequences(X_full_s, val_sig_pos, y_va_fold, seq_len, len(X_full))

        if len(X_tr_seq) < 5 or len(X_va_seq) == 0:
            print(f"  Fold {fold+1}: not enough data, skipping")
            continue

        n_pos = y_tr_seq.sum()
        n_neg = len(y_tr_seq) - n_pos
        cw    = {0: 1.0, 1: float(n_neg)/float(n_pos) if n_pos > 0 else 1.0}

        model_fold = build_model(seq_len, X_tr_seq.shape[2])
        model_fold.fit(
            X_tr_seq, y_tr_seq,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=cw,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=15,
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                  patience=7, min_lr=1e-6, verbose=0),
            ],
            verbose=0,
        )

        prob_fold = model_fold.predict(X_va_seq, verbose=0).flatten()

        # Store OOF predictions
        for j, orig_idx in enumerate(val_fold_idx):
            if j < len(prob_fold):
                oof_lstm_prob[orig_idx] = prob_fold[j]
                oof_mask[orig_idx]      = True

        fold_acc = accuracy_score(y_va_seq, (prob_fold >= 0.5).astype(int))
        print(f"  Fold {fold+1} LSTM accuracy: {fold_acc:.2%}")

        # ARIMA OOF on signal days
        for j, sig_pos in enumerate(val_sig_pos):
            arima_history = list(close_s[max(0, sig_pos-200):sig_pos])
            pred_dir, _   = predict_one(arima_history, arima_order)
            orig_idx      = val_fold_idx[j]
            oof_arima_pred[orig_idx] = pred_dir

        print(f"  Fold {fold+1} ARIMA done")
        tf.keras.backend.clear_session()

    valid = oof_mask
    print(f"\n  OOF valid samples: {valid.sum()} / {n_signals}")

    return (oof_lstm_prob[valid],
            oof_arima_pred[valid],
            y_all[valid])


# ════════════════════════════════════════════════════════
#  Train Meta Model
# ════════════════════════════════════════════════════════
def train_meta(oof_lstm_prob: np.ndarray,
               oof_arima_pred: np.ndarray,
               oof_labels: np.ndarray):
    """Train Logistic Regression as Meta Model."""
    print(f"\n{'='*55}")
    print(f"  [Stacking] Training Meta Model")
    print(f"{'='*55}")

    X_meta    = np.column_stack([oof_lstm_prob, oof_arima_pred])
    y_meta    = oof_labels
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    meta_model.fit(X_meta, y_meta)

    oof_pred = meta_model.predict(X_meta)
    oof_acc  = accuracy_score(y_meta, oof_pred)

    print(f"  OOF Accuracy : {oof_acc:.2%}")
    print(f"  LSTM  weight : {meta_model.coef_[0][0]:.4f}")
    print(f"  ARIMA weight : {meta_model.coef_[0][1]:.4f}")
    print(f"  Intercept    : {meta_model.intercept_[0]:.4f}")

    return meta_model


# ════════════════════════════════════════════════════════
#  Test set prediction
# ════════════════════════════════════════════════════════
def predict_test(meta_model,
                 lstm_prob_test: np.ndarray,
                 arima_pred_test: np.ndarray,
                 y_test: np.ndarray):
    """Apply Meta Model to test set."""
    print(f"\n{'='*55}")
    print(f"  [Stacking] Test set prediction")
    print(f"{'='*55}")

    X_meta     = np.column_stack([lstm_prob_test, arima_pred_test])
    stack_pred = meta_model.predict(X_meta)
    stack_prob = meta_model.predict_proba(X_meta)[:, 1]
    stack_acc  = accuracy_score(y_test, stack_pred)

    print(f"  Stacking Accuracy: {stack_acc:.2%}")
    print(classification_report(y_test, stack_pred,
                                target_names=["Unprofitable(0)", "Profitable(1)"],
                                digits=4))
    return stack_pred, stack_prob, stack_acc


# ════════════════════════════════════════════════════════
#  Single prediction
# ════════════════════════════════════════════════════════
def predict_one_sample(meta_model, lstm_prob: float, arima_pred: int):
    X    = np.array([[lstm_prob, arima_pred]])
    pred = int(meta_model.predict(X)[0])
    prob = float(meta_model.predict_proba(X)[0][1])
    return pred, prob


# ════════════════════════════════════════════════════════
#  Save / Load
# ════════════════════════════════════════════════════════
def save(meta_model, model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "meta_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(meta_model, f)
    print(f"  Meta model -> {path}")


def load(model_dir: str = "models"):
    path = os.path.join(model_dir, "meta_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)