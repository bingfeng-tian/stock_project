"""
lstm_model.py
LSTM model - now trained on Strategy 4 signal days only.
Predicts: will holding N days after buy signal be profitable?
"""

import os, pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)


def make_sequences(X_full: np.ndarray,
                   signal_indices: list,
                   y: np.ndarray,
                   seq_len: int,
                   full_df_len: int):
    """
    Build LSTM sequences for signal days only.

    For each signal day at index i:
      Input  = features from [i-seq_len : i] in the FULL dataset
      Output = label for that signal day

    This way LSTM still sees seq_len days of context,
    but only learns from signal-triggered days.

    Args:
        X_full         : full feature matrix (all trading days)
        signal_indices : positions of Signal_buy=1 days in X_full
        y              : labels for signal days
        seq_len        : LSTM lookback window
        full_df_len    : total number of rows in X_full
    """
    Xs, ys = [], []
    for idx, label in zip(signal_indices, y):
        if idx >= seq_len:
            Xs.append(X_full[idx - seq_len : idx])
            ys.append(label)
    return np.array(Xs), np.array(ys)


def build_model(seq_len: int, n_features: int) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        BatchNormalization(), Dropout(0.2),
        LSTM(32, return_sequences=True),
        BatchNormalization(), Dropout(0.2),
        LSTM(16, return_sequences=False), Dropout(0.2),
        Dense(16, activation="relu"), Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-4),     # 學習率 (5e-4)
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(full_df,
          signal_df,
          feature_cols: list,
          train_ratio: float = 0.8,
          seq_len: int = 30,
          epochs: int = 100,
          batch_size: int = 16):
    """
    Train LSTM on Strategy 4 signal days only.

    Key difference from v1:
      - full_df  : all trading days (for feature context)
      - signal_df: only Signal_buy=1 days (for training labels)
      - Each sequence looks back seq_len days in full_df
      - But only signal days contribute to loss

    Returns same interface as before for compatibility.
    """
    print(f"\n{'='*55}")
    print(f"  [LSTM] Training on signal days only")
    print(f"{'='*55}")

    # Full feature matrix (all days, for sequence context)
    X_full = full_df[feature_cols].values
    y_signal = signal_df["Label"].values

    # Get positions of signal days in full_df
    signal_indices = [full_df.index.get_loc(idx) for idx in signal_df.index]
    signal_indices = np.array(signal_indices)

    # Train/test split by time
    split_n      = int(len(signal_indices) * train_ratio)
    split_full   = int(len(X_full) * train_ratio)

    tr_signal_idx = signal_indices[:split_n]
    te_signal_idx = signal_indices[split_n:]
    y_tr          = y_signal[:split_n]
    y_te          = y_signal[split_n:]

    # Normalize: fit on training portion of full data only
    scaler    = MinMaxScaler()
    X_full_s  = np.zeros_like(X_full, dtype=float)
    X_full_s[:split_full]  = scaler.fit_transform(X_full[:split_full])
    X_full_s[split_full:]  = scaler.transform(X_full[split_full:])

    # Build sequences
    X_tr, y_tr_seq = make_sequences(X_full_s, tr_signal_idx, y_tr, seq_len, len(X_full))
    X_te, y_te_seq = make_sequences(X_full_s, te_signal_idx, y_te, seq_len, len(X_full))

    print(f"  Signal train samples : {len(X_tr)}")
    print(f"  Signal test  samples : {len(X_te)}")

    if len(X_tr) < 10:
        print("  WARNING: Very few training samples!")
        print("  Consider reducing hold_days or expanding date range.")

    up_pct = y_tr_seq.mean() if len(y_tr_seq) > 0 else 0
    n_pos  = y_tr_seq.sum()
    n_neg  = len(y_tr_seq) - n_pos
    cw     = {0: 1.0, 1: float(n_neg)/float(n_pos) if n_pos > 0 else 1.0}

    print(f"  Profitable rate : {up_pct:.2%}")
    print(f"  class_weight    : {cw}")

    model = build_model(seq_len, X_tr.shape[2] if len(X_tr) > 0 else len(feature_cols))
    model.summary()

    if len(X_tr) < 10:
        print("  Skipping training due to insufficient data.")
        prob   = np.full(len(y_te_seq), 0.5)
        return (model, scaler, 0.5, y_te_seq, prob,
                None, split_full, np.array([]), np.array([]))

    history = model.fit(
        X_tr, y_tr_seq,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=25,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=7, min_lr=1e-6, verbose=1),
        ],
        verbose=1,
    )

    # Best threshold
    val_size = max(1, int(len(X_tr) * 0.1))
    val_prob = model.predict(X_tr[-val_size:], verbose=0).flatten()
    val_true = y_tr_seq[-val_size:]
    best_thr, best_acc = 0.5, 0.0
    for thr in np.arange(0.30, 0.71, 0.01):
        if len(val_true) > 0:
            acc = accuracy_score(val_true, (val_prob >= thr).astype(int))
            if acc > best_acc:
                best_acc, best_thr = acc, thr
    print(f"\n  Best threshold : {best_thr:.2f}  Val accuracy: {best_acc:.2%}")

    # Test evaluation
    prob   = model.predict(X_te, verbose=0).flatten() if len(X_te) > 0 else np.array([])
    y_pred = (prob >= best_thr).astype(int) if len(prob) > 0 else np.array([])

    if len(y_te_seq) > 0 and len(y_pred) > 0:
        print(f"\n{'='*55}")
        print(f"  [LSTM] Test Evaluation (signal days only)")
        print(f"{'='*55}")
        print(f"  Accuracy: {accuracy_score(y_te_seq, y_pred):.2%}")
        print(classification_report(y_te_seq, y_pred,
                                    target_names=["Unprofitable(0)", "Profitable(1)"],
                                    digits=4))

    return (model, scaler, best_thr,
            y_te_seq, prob, history,
            split_full, np.array([]), np.array([]))


def predict(model, scaler, df,
            feature_cols: list,
            seq_len: int,
            threshold: float):
    """
    Predict if current buy signal will be profitable after N days.
    Only called when Signal_buy=1.
    """
    X_raw    = df[feature_cols].values
    X_scaled = scaler.transform(X_raw)

    if len(X_scaled) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {len(X_scaled)}")

    X_input = X_scaled[-seq_len:].reshape(1, seq_len, len(feature_cols))
    prob    = float(model.predict(X_input, verbose=0)[0][0])
    pred    = 1 if prob >= threshold else 0
    return pred, prob


def save(model, scaler, model_dir: str = "models"):
    """Save single-ticker model + scaler (legacy, kept for compatibility)."""
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "lstm_model.h5"))
    with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  LSTM model -> {model_dir}/lstm_model.h5")
    print(f"  Scaler     -> {model_dir}/scaler.pkl")


def save_multi(model, scalers_dict: dict, model_dir: str = "models"):
    """
    Save multi-ticker model + per-ticker scaler dict.

    Args:
        model        : trained Keras model (shared across tickers)
        scalers_dict : {ticker: MinMaxScaler} per-ticker scaler
        model_dir    : output directory
    """
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "lstm_model.h5"))
    with open(os.path.join(model_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers_dict, f)
    print(f"  LSTM model  -> {model_dir}/lstm_model.h5")
    print(f"  Scalers dict-> {model_dir}/scalers.pkl  ({len(scalers_dict)} tickers)")


def load(model_dir: str = "models"):
    """Load model. Returns (model, scaler) — scaler may be None for multi-ticker."""
    model = load_model(os.path.join(model_dir, "lstm_model.h5"))
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = None
    return model, scaler


def load_scalers(model_dir: str = "models") -> dict:
    """Load per-ticker scaler dict saved by save_multi()."""
    path = os.path.join(model_dir, "scalers.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run train.py with multi-ticker mode first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_with_scaler(model, scaler, df,
                        feature_cols: list,
                        seq_len: int,
                        threshold: float):
    """
    Predict using an explicitly provided scaler (for multi-ticker mode).
    Identical logic to predict(), but scaler is passed in rather than
    loaded from disk.
    """
    X_raw    = df[feature_cols].values
    X_scaled = scaler.transform(X_raw)

    if len(X_scaled) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {len(X_scaled)}")

    X_input = X_scaled[-seq_len:].reshape(1, seq_len, len(feature_cols))
    prob    = float(model.predict(X_input, verbose=0)[0][0])
    pred    = 1 if prob >= threshold else 0
    return pred, prob