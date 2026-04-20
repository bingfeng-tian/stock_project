"""
chart.py — one function = one figure = one PNG file

Chart index:
  chart01_price_bb_signals.png      Price + EMA + BB bands + buy/sell markers
  chart02_lstm_training.png         LSTM training accuracy & loss curve
  chart03_lstm_direction.png        LSTM predicted direction vs actual (signal days)
  chart04_arima_price.png           ARIMA close price forecast vs actual
  chart05_arima_direction.png       ARIMA predicted direction vs actual
  chart06_accuracy.png              Accuracy bar: LSTM / ARIMA / Stacking
  chart07_rmse.png                  RMSE bar: ARIMA per-ticker average
  chart08_mape.png                  MAPE bar: ARIMA per-ticker average
  chart09_pred_comparison.png       LSTM vs ARIMA vs Stacking prediction comparison
  chart10_stacking_prob.png         Stacking up-probability over time
  chart11_meta_weights.png          Meta Model (Logistic Regression) weights
  chart12_arima_rmse_per_ticker.png ARIMA RMSE for each ticker
  chart13_arima_mape_per_ticker.png ARIMA MAPE for each ticker
  chart14_arima_acc_per_ticker.png  ARIMA direction accuracy for each ticker
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

CHART_DIR = "charts"


# ────────────────────────────────────────────────────────
#  Internal helper
# ────────────────────────────────────────────────────────
def _save(fig, filename: str):
    os.makedirs(CHART_DIR, exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       CHART_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {out}")
    plt.close(fig)


def _bar_with_labels(ax, labels, values, colors, direction="higher"):
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.55)
    top  = max(v for v in values if v is not None)
    for bar, val in zip(bars, values):
        if val is None:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.025,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    best = (np.argmax(values) if direction == "higher" else np.argmin(values))
    bars[best].set_edgecolor("gold")
    bars[best].set_linewidth(3)
    return bars


# ════════════════════════════════════════════════════════
#  chart01 — Price + EMA + Bollinger Band + BB Signals
# ════════════════════════════════════════════════════════
def plot_price_bb_signals(df, test_start_idx: int,
                          ticker: str, start_date: str, end_date: str):
    close_all = df["Close"].squeeze()
    fig, ax   = plt.subplots(figsize=(16, 5))

    ax.plot(df.index, close_all,    "#2196F3", lw=1.2, label="Close")
    ax.plot(df.index, df["EMA_20"], "#FF9800", lw=1, ls="--", label="EMA 20")
    ax.plot(df.index, df["EMA_60"], "#9C27B0", lw=1, ls="--", label="EMA 60")
    ax.fill_between(df.index, df["BB_upper"], df["BB_lower"],
                    alpha=0.08, color="green", label="Bollinger Band")
    ax.plot(df.index, df["BB_upper"], "g-", lw=0.7)
    ax.plot(df.index, df["BB_lower"], "r-", lw=0.7)
    ax.axvline(df.index[test_start_idx], color="gray",
               ls=":", lw=1.5, label="Train/Test Split")

    buy_mask  = df["Signal_buy"]  == 1
    sell_mask = df["Signal_sell"] == 1
    if buy_mask.any():
        ax.scatter(df.index[buy_mask], close_all[buy_mask],
                   marker="^", color="#4CAF50", s=60, zorder=5,
                   label="BB Buy Signal (▲ lower touch)")
    if sell_mask.any():
        ax.scatter(df.index[sell_mask], close_all[sell_mask],
                   marker="v", color="#F44336", s=60, zorder=5,
                   label="BB Sell Signal (▼ upper touch)")

    ax.set_title(
        f"[{ticker}]  Price + EMA + Bollinger Band  "
        f"▲Buy=lower touch  ▼Sell=upper touch  ({start_date} ~ {end_date})"
    )
    ax.set_ylabel("Price (TWD)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart01_price_bb_signals.png")


# ════════════════════════════════════════════════════════
#  chart02 — LSTM Training Curve (accuracy + loss, twin axis)
# ════════════════════════════════════════════════════════
def plot_lstm_training(history_log):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(history_log.history["accuracy"],     "#2196F3", lw=1.5, label="Train Accuracy")
    ax.plot(history_log.history["val_accuracy"], "#FF5722", lw=1.5, ls="--", label="Val Accuracy")
    ax.set_ylabel("Accuracy"); ax.set_xlabel("Epoch")

    ax2 = ax.twinx()
    ax2.plot(history_log.history["loss"],     "#4CAF50", alpha=0.6, label="Train Loss")
    ax2.plot(history_log.history["val_loss"], "#F44336", alpha=0.6, ls="--", label="Val Loss")
    ax2.set_ylabel("Loss", color="gray")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    ax.set_title("LSTM Training Curve  (Accuracy & Loss)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart02_lstm_training.png")


# ════════════════════════════════════════════════════════
#  chart03 — LSTM Direction Prediction vs Actual
# ════════════════════════════════════════════════════════
def plot_lstm_direction(test_dates, y_te, lstm_preds, lstm_prob,
                        lstm_acc: float, ticker: str):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(test_dates, y_te,       "#2196F3", lw=1.5, label="Actual (1=profit)",   alpha=0.8)
    ax.plot(test_dates, lstm_preds, "#FF5722", lw=1,   label=f"LSTM Pred",          ls="--")
    ax.plot(test_dates, lstm_prob,  "#9E9E9E", lw=0.8, label="LSTM Up Probability", alpha=0.7)
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Date"); ax.set_ylabel("Label / Probability")
    ax.set_title(
        f"[{ticker}]  LSTM Direction Prediction  "
        f"Accuracy={lstm_acc:.2%}  (BB buy signal days only)"
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart03_lstm_direction.png")


# ════════════════════════════════════════════════════════
#  chart04 — ARIMA Close Price Forecast vs Actual
# ════════════════════════════════════════════════════════
def plot_arima_price(arima_actual, arima_pred,
                     arima_rmse: float, arima_mape: float,
                     arima_acc: float, ticker: str):
    idx = range(len(arima_actual))
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(idx, arima_actual, "#2196F3", lw=1.5, label="Actual Close")
    ax.plot(idx, arima_pred,   "#009688", lw=1, ls="--",
            label=f"ARIMA Forecast  RMSE={arima_rmse:.2f}  MAPE={arima_mape:.2f}%")
    ax.fill_between(idx, arima_actual, arima_pred, alpha=0.12, color="#009688")
    ax.set_ylabel("Price (TWD)"); ax.set_xlabel("Weeks")
    ax.set_title(
        f"[{ticker}]  ARIMA Close Price Forecast  "
        f"Dir.Acc={arima_acc:.2%}  RMSE={arima_rmse:.2f}  MAPE={arima_mape:.2f}%"
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart04_arima_price.png")


# ════════════════════════════════════════════════════════
#  chart05 — ARIMA Direction Prediction vs Actual
# ════════════════════════════════════════════════════════
def plot_arima_direction(arima_actual, arima_preds_dir,
                         arima_acc: float, ticker: str):
    true_dir = (arima_actual[1:] > arima_actual[:-1]).astype(int)
    idx      = range(len(true_dir))
    fig, ax  = plt.subplots(figsize=(14, 4))

    ax.plot(idx, true_dir,             "#2196F3", lw=1.5, label="Actual Direction", alpha=0.7)
    ax.plot(idx, arima_preds_dir[:-1], "#009688", lw=1,   label="ARIMA Predicted",  ls="--")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Weeks"); ax.set_ylabel("Direction (1=up, 0=down)")
    ax.set_title(f"[{ticker}]  ARIMA Direction Prediction  Accuracy={arima_acc:.2%}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart05_arima_direction.png")


# ════════════════════════════════════════════════════════
#  chart06 — Accuracy Comparison: LSTM / ARIMA / Stacking
# ════════════════════════════════════════════════════════
def plot_accuracy_comparison(lstm_acc: float, arima_acc: float, stack_acc: float,
                              ticker: str):
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["LSTM", "ARIMA", "Stacking"]
    values = [lstm_acc * 100, arima_acc * 100, stack_acc * 100]
    colors = ["#FF5722", "#009688", "#9C27B0"]

    _bar_with_labels(ax, labels, values, colors, direction="higher")
    ax.axhline(50, color="gray", lw=1, ls=":", label="random baseline 50%")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        f"[{ticker}]  BB Signal Direction Accuracy\n"
        f"(correctly predicted profitable BB buy signals)"
    )
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart06_accuracy.png")


# ════════════════════════════════════════════════════════
#  chart07 — RMSE Comparison (ARIMA vs baseline)
# ════════════════════════════════════════════════════════
def plot_rmse_comparison(arima_rmse: float, ticker: str):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.bar(["ARIMA"], [arima_rmse], color="#009688", alpha=0.85, width=0.4)
    ax.text(0, arima_rmse + arima_rmse * 0.02,
            f"{arima_rmse:.2f}", ha="center", va="bottom",
            fontsize=13, fontweight="bold")
    ax.set_ylabel("RMSE (TWD)")
    ax.set_title(
        f"[{ticker}]  ARIMA Close Price RMSE\n"
        f"(Root Mean Square Error vs actual close)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart07_rmse.png")


# ════════════════════════════════════════════════════════
#  chart08 — MAPE Comparison
# ════════════════════════════════════════════════════════
def plot_mape_comparison(arima_mape: float, ticker: str):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.bar(["ARIMA"], [arima_mape], color="#FF9800", alpha=0.85, width=0.4)
    ax.text(0, arima_mape + arima_mape * 0.02,
            f"{arima_mape:.2f}%", ha="center", va="bottom",
            fontsize=13, fontweight="bold")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(
        f"[{ticker}]  ARIMA Close Price MAPE\n"
        f"(Mean Absolute Percentage Error)"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart08_mape.png")


# ════════════════════════════════════════════════════════
#  chart09 — LSTM vs ARIMA vs Stacking Prediction Comparison
# ════════════════════════════════════════════════════════
def plot_pred_comparison(test_dates, y_te, lstm_preds, arima_preds_dir,
                         stack_pred, lstm_acc, arima_acc, stack_acc,
                         ticker: str):
    n = min(len(y_te), len(lstm_preds), len(arima_preds_dir), len(stack_pred))
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(test_dates[:n], y_te[:n],            "#2196F3", lw=2,   label="Actual",                    alpha=0.8)
    ax.plot(test_dates[:n], lstm_preds[:n],      "#FF5722", lw=1,   label=f"LSTM  ({lstm_acc:.2%})",   ls="--", alpha=0.8)
    ax.plot(test_dates[:n], arima_preds_dir[:n], "#009688", lw=1,   label=f"ARIMA ({arima_acc:.2%})",  ls=":",  alpha=0.8)
    ax.plot(test_dates[:n], stack_pred[:n],      "#9C27B0", lw=1.5, label=f"Stack ({stack_acc:.2%})")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Date"); ax.set_ylabel("Prediction (1=up, 0=down)")
    ax.set_title(
        f"[{ticker}]  LSTM vs ARIMA vs Stacking  (BB signal days)\n"
        f"LSTM={lstm_acc:.2%}  ARIMA={arima_acc:.2%}  Stack={stack_acc:.2%}"
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart09_pred_comparison.png")


# ════════════════════════════════════════════════════════
#  chart10 — Stacking Up Probability Over Time
# ════════════════════════════════════════════════════════
def plot_stacking_prob(test_dates, stack_prob, stack_acc: float, ticker: str):
    n   = min(len(test_dates), len(stack_prob))
    fig, ax = plt.subplots(figsize=(14, 4))

    sp = stack_prob[:n]
    ax.plot(test_dates[:n], sp, "#9C27B0", lw=1.2, label="Stacking Up Probability")
    ax.fill_between(test_dates[:n], sp, 0.5,
                    where=(sp >= 0.5), alpha=0.2, color="#4CAF50", label="Predicted Up")
    ax.fill_between(test_dates[:n], sp, 0.5,
                    where=(sp <  0.5), alpha=0.2, color="#F44336", label="Predicted Down")
    ax.axhline(0.5, color="gray", lw=1, ls=":")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Date"); ax.set_ylabel("Up Probability")
    ax.set_title(
        f"[{ticker}]  Stacking Up Probability  Accuracy={stack_acc:.2%}\n"
        f"(1 = BB reversion UP expected  |  0 = DOWN expected)"
    )
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart10_stacking_prob.png")


# ════════════════════════════════════════════════════════
#  chart11 — Meta Model Weights (Logistic Regression)
# ════════════════════════════════════════════════════════
def plot_meta_weights(meta_model, ticker: str):
    coefs = [meta_model.coef_[0][0], meta_model.coef_[0][1]]
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = ["#FF5722" if c >= 0 else "#2196F3" for c in coefs]
    bars   = ax.bar(["LSTM weight", "ARIMA weight"], coefs,
                    color=colors, alpha=0.85, width=0.45)
    for bar, val in zip(bars, coefs):
        offset = abs(val) * 0.04 + 0.005
        va     = "bottom" if val >= 0 else "top"
        y      = bar.get_height() + (offset if val >= 0 else -offset)
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{val:.4f}", ha="center", va=va,
                fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Coefficient")
    ax.set_title(
        f"[{ticker}]  Stacking Meta Model Weights\n"
        f"Positive=bullish  Negative=bearish  "
        f"Intercept={meta_model.intercept_[0]:.4f}"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart11_meta_weights.png")


# ════════════════════════════════════════════════════════
#  chart12 — ARIMA RMSE per Ticker
# ════════════════════════════════════════════════════════
def plot_arima_rmse_per_ticker(ticker_stats: list):
    if not ticker_stats:
        return
    tickers   = [s["ticker"].replace(".TW", "") for s in ticker_stats]
    rmse_vals = [s["arima_rmse"] for s in ticker_stats]
    avg       = float(np.mean(rmse_vals))

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(tickers, rmse_vals, color="#009688", alpha=0.85, width=0.6)
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rmse_vals) * 0.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    best = int(np.argmin(rmse_vals))
    bars[best].set_edgecolor("gold"); bars[best].set_linewidth(2.5)
    ax.axhline(avg, color="red", lw=1.2, ls="--", label=f"avg = {avg:.1f}")
    ax.set_xlabel("Ticker"); ax.set_ylabel("RMSE (TWD)")
    ax.set_title("ARIMA Close Price RMSE per Ticker  ↓ lower is better")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart12_arima_rmse_per_ticker.png")


# ════════════════════════════════════════════════════════
#  chart13 — ARIMA MAPE per Ticker
# ════════════════════════════════════════════════════════
def plot_arima_mape_per_ticker(ticker_stats: list):
    if not ticker_stats:
        return
    tickers   = [s["ticker"].replace(".TW", "") for s in ticker_stats]
    mape_vals = [s["arima_mape"] for s in ticker_stats]
    avg       = float(np.mean(mape_vals))

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(tickers, mape_vals, color="#FF9800", alpha=0.85, width=0.6)
    for bar, val in zip(bars, mape_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(mape_vals) * 0.02,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    best = int(np.argmin(mape_vals))
    bars[best].set_edgecolor("gold"); bars[best].set_linewidth(2.5)
    ax.axhline(avg, color="red", lw=1.2, ls="--", label=f"avg = {avg:.2f}%")
    ax.set_xlabel("Ticker"); ax.set_ylabel("MAPE (%)")
    ax.set_title("ARIMA Close Price MAPE per Ticker  ↓ lower is better")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart13_arima_mape_per_ticker.png")


# ════════════════════════════════════════════════════════
#  chart14 — ARIMA Direction Accuracy per Ticker
# ════════════════════════════════════════════════════════
def plot_arima_acc_per_ticker(ticker_stats: list):
    if not ticker_stats:
        return
    tickers  = [s["ticker"].replace(".TW", "") for s in ticker_stats]
    acc_vals = [s["arima_acc"] * 100 for s in ticker_stats]
    avg      = float(np.mean(acc_vals))

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(tickers, acc_vals, color="#5C6BC0", alpha=0.85, width=0.6)
    for bar, val in zip(bars, acc_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    best = int(np.argmax(acc_vals))
    bars[best].set_edgecolor("gold"); bars[best].set_linewidth(2.5)
    ax.axhline(50,  color="gray", lw=1, ls=":",  label="random baseline 50%")
    ax.axhline(avg, color="red",  lw=1.2, ls="--", label=f"avg = {avg:.1f}%")
    ax.set_xlabel("Ticker"); ax.set_ylabel("Direction Accuracy (%)")
    ax.set_title("ARIMA Direction Accuracy per Ticker  ↑ higher is better")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart14_arima_acc_per_ticker.png")


# ════════════════════════════════════════════════════════
#  Convenience: plot all per-ticker charts at once
# ════════════════════════════════════════════════════════
def plot_arima_metrics_by_ticker(ticker_stats: list):
    """Called from train.py after multi-ticker training."""
    print(f"\n{'='*55}")
    print(f"  [Chart] Generating per-ticker metric charts")
    print(f"{'='*55}")
    plot_arima_rmse_per_ticker(ticker_stats)
    plot_arima_mape_per_ticker(ticker_stats)
    plot_arima_acc_per_ticker(ticker_stats)
    print(f"  Charts 12–14 saved to ./{CHART_DIR}/")
