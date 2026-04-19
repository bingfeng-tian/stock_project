"""
train.py - Main training script (Bollinger Band primary signal)

TRAINING LOGIC:
  Model learns from BB buy signal days only.
  BB buy signal = close <= BB_lower (price touches lower band → oversold)

  Label = "will holding N days after this BB signal be profitable?"
          1 = price rises (mean reversion succeeds)
          0 = price falls (mean reversion fails)

  This aligns with the BB mean reversion trading logic:
    - Enter when BB buy signal triggers (close touches lower band)
    - Hold for N days (or until BB sell signal at upper band)
    - Goal: predict if this BB trade will result in a gain

Flow:
  1. data_loader  -> download + compute features (BB primary signals)
  2. data_loader  -> filter BB signal days + compute hold-N-day labels
  3. lstm_model   -> train on BB signal days only
  4. arima_model  -> rolling forecast on test set (all days)
  5. stacking     -> OOF + Meta Model
  6. Save models
  7. chart        -> generate charts
"""

import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, mean_squared_error

import data_loader
import lstm_model
import arima_model
import stacking
import chart

# ════════════════════════════════════════════════════════
#  SETTINGS
# ════════════════════════════════════════════════════════
TICKER      = "2330.TW"         # 訓練標的
START_DATE  = "2018-01-01"      # 訓練開始日期
END_DATE    = None              # 訓練結束日期 (None = 今天)    
TRAIN_RATIO = 0.8               # 訓練/測試比例(0.8 = 80%%訓練 / 20%測試)
SEQ_LEN     = 30                # LSTM lookback window (days of context)
EPOCHS      = 100               # LSTM 迭代次數
BATCH_SIZE  = 16       # smaller batch for fewer signal samples
ARIMA_ORDER = (2, 1, 2)
N_FOLDS     = 5
HOLD_DAYS   = 5        # hold N days after buy signal
MODEL_DIR   = "models"


def calc_rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(
        np.array(actual, dtype=float),
        np.array(predicted, dtype=float)
    )))

def calc_mape(actual, predicted):
    a = np.array(actual,    dtype=float)
    p = np.array(predicted, dtype=float)
    mask = np.abs(a) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def save_meta(df, best_thr, end_label):
    os.makedirs(MODEL_DIR, exist_ok=True)
    meta = {
        "ticker"         : TICKER,
        "train_start"    : START_DATE,
        "train_end"      : end_label,
        "seq_len"        : SEQ_LEN,
        "hold_days"      : HOLD_DAYS,
        "feature_cols"   : data_loader.FEATURE_COLS,
        "best_threshold" : float(best_thr),
        "arima_order"    : list(ARIMA_ORDER),
        "saved_at"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_mode"     : "bb_signal_mean_reversion",  # BB primary signal
        "signal_logic"   : "buy=close<=BB_lower, sell=close>=BB_upper",
    }
    path = os.path.join(MODEL_DIR, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Config -> {path}")


def main():
    end_label = END_DATE if END_DATE else datetime.today().strftime("%Y-%m-%d")

    # ── 1. Download + features ───────────────────────────
    raw = data_loader.download_data(TICKER, START_DATE, END_DATE)
    full_df = data_loader.compute_features(raw)
    print(f"\n  Total samples : {len(full_df)}")
    print(f"  Range         : {full_df.index[0].date()} ~ {full_df.index[-1].date()}")

    # ── 2. Signal-based labels ───────────────────────────
    print(f"\n  Hold days after signal : {HOLD_DAYS}")
    signal_df, full_df = data_loader.compute_signal_labels(full_df, HOLD_DAYS)

    if len(signal_df) < 20:
        print(f"\n  WARNING: Only {len(signal_df)} signal samples found.")
        print(f"  Consider adjusting START_DATE or HOLD_DAYS.")

    split_full = int(len(full_df) * TRAIN_RATIO)

    # ── 3. Train LSTM on signal days ─────────────────────
    (model, scaler, best_thr,
     y_te, lstm_prob, history_log,
     split_idx, _, _) = lstm_model.train(
        full_df, signal_df,
        data_loader.FEATURE_COLS,
        TRAIN_RATIO, SEQ_LEN, EPOCHS, BATCH_SIZE,
    )
    lstm_preds     = (lstm_prob >= best_thr).astype(int) if len(lstm_prob) > 0 else np.array([])
    test_start_idx = split_full

    # ── 4. ARIMA rolling forecast ────────────────────────
    close_s = full_df["Close"].squeeze()
    arima_preds_dir, arima_preds_val, arima_actual_val, arima_acc = \
        arima_model.rolling_forecast(
            close_s.iloc[:test_start_idx],
            close_s.iloc[test_start_idx:],
            ARIMA_ORDER,
        )

    # ── 5. Stacking ──────────────────────────────────────
    # For stacking we align ARIMA predictions with signal days
    # Get test signal indices
    signal_test = signal_df[signal_df.index >= full_df.index[test_start_idx]]
    signal_test_pos = [
        full_df.index.get_loc(idx) - test_start_idx
        for idx in signal_test.index
        if full_df.index.get_loc(idx) - test_start_idx < len(arima_preds_dir)
    ]

    if len(lstm_prob) > 0 and len(signal_test_pos) > 0:
        # Align: take ARIMA predictions at signal days
        arima_at_signals = arima_preds_dir[signal_test_pos[:len(lstm_prob)]]
        y_te_aligned     = y_te[:len(arima_at_signals)]
        lstm_prob_aligned = lstm_prob[:len(arima_at_signals)]

        # OOF for meta model
        oof_lstm, oof_arima, oof_labels = stacking.generate_oof(
            full_df, signal_df,
            data_loader.FEATURE_COLS,
            split_full, SEQ_LEN, N_FOLDS,
            ARIMA_ORDER, EPOCHS, BATCH_SIZE,
            HOLD_DAYS,
        )

        if len(oof_lstm) > 0:
            meta_model = stacking.train_meta(oof_lstm, oof_arima, oof_labels)
            stack_pred, stack_prob, stack_acc = stacking.predict_test(
                meta_model, lstm_prob_aligned, arima_at_signals, y_te_aligned
            )
        else:
            print("\n  Stacking skipped: insufficient OOF samples")
            meta_model = None
            stack_pred = lstm_preds
            stack_prob = lstm_prob
            stack_acc  = accuracy_score(y_te, lstm_preds) if len(y_te) > 0 else 0.0
    else:
        print("\n  Stacking skipped: insufficient signal samples")
        meta_model = None
        stack_pred = np.array([])
        stack_prob = np.array([])
        stack_acc  = 0.0

    # ── 6. Save ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  [Save] Exporting models -> ./{MODEL_DIR}/")
    print(f"{'='*55}")
    lstm_model.save(model, scaler, MODEL_DIR)
    arima_model.save(ARIMA_ORDER, close_s, MODEL_DIR)
    if meta_model is not None:
        stacking.save(meta_model, MODEL_DIR)
    save_meta(full_df, best_thr, end_label)

    # ── 7. Metrics ───────────────────────────────────────
    lstm_acc   = accuracy_score(y_te, lstm_preds) if len(y_te) > 0 and len(lstm_preds) > 0 else 0.0
    arima_rmse = calc_rmse(arima_actual_val, arima_preds_val)
    arima_mape = calc_mape(arima_actual_val, arima_preds_val)

    print(f"""
{'='*55}
  Final Metrics Summary
{'='*55}
  Training mode : Strategy 4 signal days only
  Hold days     : {HOLD_DAYS} days after buy signal
  Signal samples: {len(signal_df)} total  (train={int(len(signal_df)*TRAIN_RATIO)}  test={len(signal_df)-int(len(signal_df)*TRAIN_RATIO)})

  +------------------+----------+----------+----------+
  |   Metric         |   LSTM   |  ARIMA   | Stacking |
  +------------------+----------+----------+----------+
  |  Accuracy        | {lstm_acc:>6.2%}   | {arima_acc:>6.2%}   | {stack_acc:>6.2%}   |
  |  RMSE (close)    |   N/A    | {arima_rmse:>8.4f} |    N/A   |
  |  MAPE (close)    |   N/A    | {arima_mape:>8.4f} |    N/A   |
  +------------------+----------+----------+----------+
  LSTM/Stacking Accuracy = % of profitable signals predicted correctly
  ARIMA RMSE/MAPE        = close price prediction error
    """)

    # ── 8. Charts ────────────────────────────────────────
    if history_log is not None and len(arima_preds_val) > 0:
        chart.plot_all(
            full_df, test_start_idx, y_te,
            lstm_preds, lstm_prob,
            arima_preds_dir, arima_preds_val, arima_actual_val,
            stack_pred, stack_prob,
            np.array([]), np.array([]),
            history_log, meta_model,
            lstm_acc, arima_acc, stack_acc,
            0.0, arima_rmse,
            0.0, arima_mape,
            TICKER, START_DATE, end_label,
        )

    print(f"\nTraining complete!")
    print(f"  Signal logic  : BB mean reversion (buy at lower band, sell at upper band)")
    print(f"  Model predicts: will BB buy signal be profitable after {HOLD_DAYS} days?")
    print(f"  Run predict.py to check today's BB signal\n")


if __name__ == "__main__":
    main()