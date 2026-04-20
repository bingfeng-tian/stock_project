"""
train.py - Multi-ticker BB Signal training (weekly data)

TRAINING LOGIC:
  - Download weekly (1wk) OHLCV for each ticker
  - Compute BB primary signals per ticker
  - Build per-ticker sequences with per-ticker MinMaxScaler
  - Combine all training sequences → train ONE shared LSTM
  - ARIMA runs per ticker → directional predictions collected at signal days
  - Stacking OOF on combined sequences → Meta Model
  - Save: lstm_model.h5 / scalers.pkl / arima_multi.pkl / meta_model.pkl / meta.json

Usage:
  python train.py
"""

import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import data_loader
import lstm_model
import arima_model
import stacking

tf.random.set_seed(42)
np.random.seed(42)

# ════════════════════════════════════════════════════════
#  SETTINGS
# ════════════════════════════════════════════════════════
TICKERS = [
    "2330.TW",   # 台積電
    "2317.TW",   # 鴻海
    "2454.TW",   # 聯發科
    "3711.TW",   # 日月光投控
    "2383.TW",   # 台光電
    "2382.TW",   # 廣達
    "2303.TW",   # 聯電
    "3017.TW",   # 奇鋐
    "3037.TW",   # 欣興
    "2357.TW",   # 華碩
]

INTERVAL    = "1wk"         # 週資料
START_DATE  = "2018-01-01"
END_DATE    = None
TRAIN_RATIO = 0.8
SEQ_LEN     = 20            # 20週 ≈ 5個月的歷史上下文
EPOCHS      = 100
BATCH_SIZE  = 32            # 合併多標的後樣本多，batch可以大一點
ARIMA_ORDER = (2, 1, 2)
N_FOLDS     = 5
HOLD_DAYS   = 4             # 持有4週 (約1個月)
MODEL_DIR   = "models"


# ════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════
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


def _train_lstm_on_sequences(X_tr, y_tr, X_te, y_te):
    """
    Train LSTM directly on pre-built (multi-ticker) sequences.
    Returns (model, best_thr, test_prob, history).
    """
    print(f"\n{'='*55}")
    print(f"  [LSTM] Training on combined sequences")
    print(f"  Train: {len(X_tr)}  Test: {len(X_te)}")
    print(f"{'='*55}")

    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    cw    = {0: 1.0, 1: float(n_neg) / float(n_pos) if n_pos > 0 else 1.0}
    print(f"  Profitable rate : {y_tr.mean():.2%}")
    print(f"  class_weight    : {cw}")

    model = lstm_model.build_model(SEQ_LEN, X_tr.shape[2])
    model.summary()

    if len(X_tr) < 10:
        print("  WARNING: Not enough training sequences.")
        prob = np.full(len(X_te), 0.5)
        return model, 0.5, prob, None

    history = model.fit(
        X_tr, y_tr,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=25,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=7, min_lr=1e-6, verbose=1),
        ],
        verbose=1,
    )

    # Best threshold on validation tail
    val_size = max(1, int(len(X_tr) * 0.1))
    val_prob = model.predict(X_tr[-val_size:], verbose=0).flatten()
    val_true = y_tr[-val_size:]
    best_thr, best_acc = 0.5, 0.0
    for thr in np.arange(0.30, 0.71, 0.01):
        if len(val_true) > 0:
            acc = accuracy_score(val_true, (val_prob >= thr).astype(int))
            if acc > best_acc:
                best_acc, best_thr = acc, thr
    print(f"\n  Best threshold : {best_thr:.2f}  Val accuracy: {best_acc:.2%}")

    # Test evaluation
    prob = model.predict(X_te, verbose=0).flatten() if len(X_te) > 0 else np.array([])
    if len(y_te) > 0 and len(prob) > 0:
        y_pred = (prob >= best_thr).astype(int)
        print(f"  Test Accuracy  : {accuracy_score(y_te, y_pred):.2%}")

    return model, best_thr, prob, history


def save_meta(tickers, best_thr, end_label):
    os.makedirs(MODEL_DIR, exist_ok=True)
    meta = {
        "tickers"        : tickers,
        "train_start"    : START_DATE,
        "train_end"      : end_label,
        "interval"       : INTERVAL,
        "seq_len"        : SEQ_LEN,
        "hold_days"      : HOLD_DAYS,
        "feature_cols"   : data_loader.FEATURE_COLS,
        "best_threshold" : float(best_thr),
        "arima_order"    : list(ARIMA_ORDER),
        "saved_at"       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_mode"     : "multi_ticker_bb_weekly",
        "signal_logic"   : "buy=close<=BB_lower, sell=close>=BB_upper",
    }
    path = os.path.join(MODEL_DIR, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  Config -> {path}")


# ════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════
def main():
    end_label = END_DATE if END_DATE else datetime.today().strftime("%Y-%m-%d")

    print(f"\n{'='*55}")
    print(f"  Multi-Ticker BB Signal Training (Weekly)")
    print(f"  Tickers  : {len(TICKERS)} stocks")
    print(f"  Interval : {INTERVAL}  |  SEQ_LEN={SEQ_LEN}  |  HOLD={HOLD_DAYS}w")
    print(f"  Period   : {START_DATE} ~ {end_label}")
    print(f"{'='*55}")

    # ── Per-ticker data collection ───────────────────────
    all_X_tr,       all_y_tr       = [], []
    all_X_te,       all_y_te       = [], []
    all_arima_tr_p, all_arima_te_p = [], []
    scalers_dict = {}
    arima_dict   = {}
    ticker_stats = []
    valid_tickers = []

    # For chart01 / chart04 / chart05: capture first valid ticker
    first_ticker_df     = None
    first_ticker_name   = None
    first_ticker_split  = None
    first_arima_actual  = None
    first_arima_pred    = None
    first_arima_dir     = None
    first_arima_acc_val = None

    for ticker in TICKERS:
        print(f"\n{'─'*55}")
        print(f"  [{ticker}] Downloading & processing...")
        print(f"{'─'*55}")
        try:
            raw = data_loader.download_data(ticker, START_DATE, END_DATE, INTERVAL)
            if len(raw) < SEQ_LEN + 20:
                print(f"  SKIP: Not enough bars ({len(raw)})")
                continue

            full_df = data_loader.compute_features(raw)
            signal_df, full_df = data_loader.compute_signal_labels(full_df, HOLD_DAYS)

            if len(signal_df) < 8:
                print(f"  SKIP: Too few signal days ({len(signal_df)})")
                continue

            X_full     = full_df[data_loader.FEATURE_COLS].values
            split_full = int(len(full_df)   * TRAIN_RATIO)
            split_n    = int(len(signal_df) * TRAIN_RATIO)
            close_s    = full_df["Close"].squeeze()

            # Per-ticker MinMaxScaler (prices differ across tickers)
            scaler   = MinMaxScaler()
            X_scaled = np.zeros_like(X_full, dtype=float)
            X_scaled[:split_full] = scaler.fit_transform(X_full[:split_full])
            X_scaled[split_full:] = scaler.transform(X_full[split_full:])
            scalers_dict[ticker]  = scaler

            # Signal positions in full_df
            sig_idx = np.array([full_df.index.get_loc(i) for i in signal_df.index])
            y_sig   = signal_df["Label"].values

            tr_sig_idx = sig_idx[:split_n];  y_tr = y_sig[:split_n]
            te_sig_idx = sig_idx[split_n:];   y_te = y_sig[split_n:]

            # Build LSTM sequences
            X_tr_seq, y_tr_seq = lstm_model.make_sequences(
                X_scaled, tr_sig_idx, y_tr, SEQ_LEN, len(X_full))
            X_te_seq, y_te_seq = lstm_model.make_sequences(
                X_scaled, te_sig_idx, y_te, SEQ_LEN, len(X_full))

            if len(X_tr_seq) >= 3:
                all_X_tr.append(X_tr_seq); all_y_tr.append(y_tr_seq)
            if len(X_te_seq) >= 1:
                all_X_te.append(X_te_seq); all_y_te.append(y_te_seq)

            # ── ARIMA rolling forecast ─────────────────────
            arima_dir_te, arima_pred_val, arima_actual_val, arima_acc_te = \
                arima_model.rolling_forecast(
                    close_s.iloc[:split_full],
                    close_s.iloc[split_full:],
                    ARIMA_ORDER,
                )
            arima_rmse_tk = calc_rmse(arima_actual_val, arima_pred_val)
            arima_mape_tk = calc_mape(arima_actual_val, arima_pred_val)

            # Keep first valid ticker for per-ticker charts
            if first_ticker_df is None:
                first_ticker_df     = full_df
                first_ticker_name   = ticker
                first_ticker_split  = split_full
                first_arima_actual  = list(arima_actual_val)
                first_arima_pred    = list(arima_pred_val)
                first_arima_dir     = arima_dir_te
                first_arima_acc_val = arima_acc_te

            # ARIMA at TEST signal days
            te_offsets = [idx - split_full for idx in te_sig_idx
                          if 0 <= idx - split_full < len(arima_dir_te)]
            if te_offsets:
                all_arima_te_p.append(arima_dir_te[te_offsets])

            # ARIMA at TRAIN signal days (for OOF later)
            tr_arima_preds = []
            for sig_pos in tr_sig_idx:
                hist = list(close_s.iloc[max(0, sig_pos - 200):sig_pos])
                if len(hist) >= 3:
                    p, _ = arima_model.predict_one(hist, ARIMA_ORDER)
                else:
                    p = 1
                tr_arima_preds.append(p)
            if tr_arima_preds:
                all_arima_tr_p.append(np.array(tr_arima_preds[:len(X_tr_seq)]))

            # Save per-ticker ARIMA state
            arima_dict[ticker] = {
                "order"        : list(ARIMA_ORDER),
                "history_tail" : list(close_s.values[-200:]),
            }

            stat = {
                "ticker"      : ticker,
                "bars"        : len(full_df),
                "signals"     : len(signal_df),
                "tr_seqs"     : len(X_tr_seq),
                "te_seqs"     : len(X_te_seq),
                "sig_rate"    : len(signal_df) / len(full_df),
                "profit_rate" : float(y_sig.mean()),
                "arima_acc"   : arima_acc_te,
                "arima_rmse"  : arima_rmse_tk,
                "arima_mape"  : arima_mape_tk,
            }
            ticker_stats.append(stat)
            valid_tickers.append(ticker)
            print(f"  ✓ signals={len(signal_df)}  tr_seqs={len(X_tr_seq)}  "
                  f"te_seqs={len(X_te_seq)}  profit={y_sig.mean():.1%}")

        except Exception as e:
            print(f"  ERROR {ticker}: {e}")
            continue

    if not all_X_tr:
        print("\nERROR: No valid training data. Exiting.")
        return

    # ── Combined arrays ──────────────────────────────────
    X_tr_all = np.concatenate(all_X_tr)
    y_tr_all = np.concatenate(all_y_tr)
    X_te_all = np.concatenate(all_X_te) if all_X_te else np.array([]).reshape(0, SEQ_LEN, len(data_loader.FEATURE_COLS))
    y_te_all = np.concatenate(all_y_te) if all_y_te else np.array([])
    arima_te_all = np.concatenate(all_arima_te_p) if all_arima_te_p else np.array([])
    arima_tr_all = np.concatenate(all_arima_tr_p) if all_arima_tr_p else np.zeros(len(X_tr_all))

    # Align arima_tr_all length
    if len(arima_tr_all) != len(X_tr_all):
        arima_tr_all = np.zeros(len(X_tr_all))

    print(f"\n{'='*65}")
    print(f"  Combined Dataset Summary")
    print(f"{'='*65}")
    print(f"  {'Ticker':<12} {'Bars':>5} {'Sig':>5} {'Tr':>4} {'Te':>4} "
          f"{'Profit':>7} {'Acc':>7} {'RMSE':>8} {'MAPE':>7}")
    print(f"  {'─'*65}")
    for s in ticker_stats:
        print(f"  {s['ticker']:<12} {s['bars']:>5} {s['signals']:>5} "
              f"{s['tr_seqs']:>4} {s['te_seqs']:>4} "
              f"{s['profit_rate']:>6.1%} {s['arima_acc']:>6.1%} "
              f"{s['arima_rmse']:>8.2f} {s['arima_mape']:>6.2f}%")
    print(f"  {'─'*65}")
    avg_rmse = float(np.mean([s['arima_rmse'] for s in ticker_stats]))
    avg_mape = float(np.mean([s['arima_mape'] for s in ticker_stats]))
    print(f"  {'TOTAL/AVG':<12} {'':>5} {'':>5} {len(X_tr_all):>4} {len(X_te_all):>4} "
          f"{'':>7} {'':>7} {avg_rmse:>8.2f} {avg_mape:>6.2f}%")

    # ── Train LSTM ───────────────────────────────────────
    model, best_thr, lstm_prob_te, history_log = _train_lstm_on_sequences(
        X_tr_all, y_tr_all, X_te_all, y_te_all
    )
    lstm_preds_te = (lstm_prob_te >= best_thr).astype(int) if len(lstm_prob_te) > 0 else np.array([])
    lstm_acc = accuracy_score(y_te_all, lstm_preds_te) if len(y_te_all) > 0 and len(lstm_preds_te) > 0 else 0.0

    # ARIMA combined accuracy on test
    if len(arima_te_all) > 0 and len(y_te_all) > 0:
        n_align    = min(len(arima_te_all), len(y_te_all))
        arima_acc  = accuracy_score(y_te_all[:n_align], arima_te_all[:n_align])
    else:
        arima_acc  = 0.0

    # ── Stacking（無 OOF，直接用訓練集預測訓練 Meta）───────
    # 移除 K-Fold 交叉訓練，改用主 LSTM 對訓練集的 in-sample 預測
    # 直接訓練 Meta Model，速度快且不重複訓練多個 LSTM
    print(f"\n{'='*55}")
    print(f"  [Stacking] Training Meta (in-sample, no cross-val)")
    print(f"{'='*55}")

    stack_acc  = 0.0
    meta_model = None
    stack_pred = lstm_preds_te
    stack_prob = lstm_prob_te

    if len(X_tr_all) >= 5 and len(arima_tr_all) == len(X_tr_all):
        # Get LSTM probability on training set (no re-training)
        lstm_prob_tr = model.predict(X_tr_all, verbose=0).flatten()

        # Align ARIMA training predictions
        n_tr_align   = min(len(lstm_prob_tr), len(arima_tr_all), len(y_tr_all))
        meta_model   = stacking.train_meta(
            lstm_prob_tr[:n_tr_align],
            arima_tr_all[:n_tr_align],
            y_tr_all[:n_tr_align],
        )

        # Evaluate on test
        n_te = min(len(lstm_prob_te), len(arima_te_all), len(y_te_all))
        if n_te > 0:
            stack_pred, stack_prob, stack_acc = stacking.predict_test(
                meta_model,
                lstm_prob_te[:n_te],
                arima_te_all[:n_te],
                y_te_all[:n_te],
            )
        else:
            print("\n  Stacking test: no aligned test samples")
    else:
        print("\n  Stacking skipped: insufficient training samples")

    # ── Save ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  [Save] Exporting models -> ./{MODEL_DIR}/")
    print(f"{'='*55}")
    lstm_model.save_multi(model, scalers_dict, MODEL_DIR)
    arima_model.save_multi(arima_dict, MODEL_DIR)
    if meta_model is not None:
        stacking.save(meta_model, MODEL_DIR)
    save_meta(valid_tickers, best_thr, end_label)

    # ── Final Metrics ────────────────────────────────────
    print(f"""
{'='*65}
  Final Metrics Summary  (Multi-Ticker Weekly BB)
{'='*65}
  Tickers  : {len(valid_tickers)}  {valid_tickers}
  Interval : {INTERVAL}  |  Hold={HOLD_DAYS} weeks
  Seqs     : train={len(X_tr_all)}  test={len(X_te_all)}

  +--------------------+----------+----------+----------+
  |   Metric           |   LSTM   |  ARIMA   | Stacking |
  +--------------------+----------+----------+----------+
  |  Accuracy          | {lstm_acc:>6.2%}   | {arima_acc:>6.2%}   | {stack_acc:>6.2%}   |
  |  RMSE (avg/ticker) |    N/A   | {avg_rmse:>8.3f} |    N/A   |
  |  MAPE (avg/ticker) |    N/A   | {avg_mape:>7.3f}% |    N/A   |
  +--------------------+----------+----------+----------+
  Accuracy = % of BB buy signals correctly predicted as profitable
  RMSE/MAPE = ARIMA close price prediction error (per-ticker avg)
    """)

    # ── Generate charts ──────────────────────────────────
    import chart as chart_mod
    print(f"\n{'='*55}")
    print(f"  [Chart] Generating output charts -> ./charts/")
    print(f"{'='*55}")

    # chart01 — Price + EMA + Bollinger Band + signals (first ticker)
    if first_ticker_df is not None:
        chart_mod.plot_price_bb_signals(
            first_ticker_df,
            first_ticker_split,
            first_ticker_name,
            START_DATE,
            end_label,
        )

    # chart02 — LSTM training curve
    if history_log is not None:
        chart_mod.plot_lstm_training(history_log)

    # chart03 — LSTM direction prediction on test set
    if len(y_te_all) > 0 and len(lstm_preds_te) > 0:
        te_indices = np.arange(len(y_te_all))
        n03 = min(len(te_indices), len(lstm_preds_te), len(lstm_prob_te))
        chart_mod.plot_lstm_direction(
            te_indices[:n03],
            y_te_all[:n03],
            lstm_preds_te[:n03],
            lstm_prob_te[:n03],
            lstm_acc,
            "Multi-Ticker",
        )

    # chart04 — ARIMA close price forecast vs actual (first ticker)
    if first_arima_actual is not None and len(first_arima_actual) > 1:
        chart_mod.plot_arima_price(
            first_arima_actual,
            first_arima_pred,
            calc_rmse(first_arima_actual, first_arima_pred),
            calc_mape(first_arima_actual, first_arima_pred),
            first_arima_acc_val,
            first_ticker_name,
        )

    # chart05 — ARIMA direction prediction vs actual (first ticker)
    if first_arima_actual is not None and len(first_arima_actual) > 2:
        chart_mod.plot_arima_direction(
            np.array(first_arima_actual),
            first_arima_dir,
            first_arima_acc_val,
            first_ticker_name,
        )

    # chart06 — Accuracy comparison: LSTM / ARIMA / Stacking
    chart_mod.plot_accuracy_comparison(lstm_acc, arima_acc, stack_acc, "Multi-Ticker")

    # chart07 — RMSE bar (average across tickers)
    chart_mod.plot_rmse_comparison(avg_rmse, "Multi-Ticker Avg")

    # chart08 — MAPE bar (average across tickers)
    chart_mod.plot_mape_comparison(avg_mape, "Multi-Ticker Avg")

    # chart09 — LSTM vs ARIMA vs Stacking prediction comparison
    if (len(y_te_all) > 0 and len(lstm_preds_te) > 0
            and len(arima_te_all) > 0 and len(stack_pred) > 0):
        te_indices = np.arange(len(y_te_all))
        n09 = min(len(te_indices), len(lstm_preds_te),
                  len(arima_te_all), len(stack_pred), len(y_te_all))
        chart_mod.plot_pred_comparison(
            te_indices[:n09],
            y_te_all[:n09],
            lstm_preds_te[:n09],
            arima_te_all[:n09],
            stack_pred[:n09],
            lstm_acc, arima_acc, stack_acc,
            "Multi-Ticker",
        )

    # chart10 — Stacking up probability over time
    if len(stack_prob) > 0 and len(y_te_all) > 0:
        te_indices = np.arange(len(stack_prob))
        chart_mod.plot_stacking_prob(
            te_indices, stack_prob, stack_acc, "Multi-Ticker"
        )

    # chart11 — Meta Model weights
    if meta_model is not None:
        chart_mod.plot_meta_weights(meta_model, "Multi-Ticker")

    # chart12–14 — Per-ticker ARIMA metrics
    chart_mod.plot_arima_metrics_by_ticker(ticker_stats)

    print(f"\nTraining complete!")
    print(f"  Run: python predict.py 2330.TW  (or any trained ticker)")
    print(f"  Interval: {INTERVAL}  |  Hold: {HOLD_DAYS} weeks\n")


if __name__ == "__main__":
    main()
