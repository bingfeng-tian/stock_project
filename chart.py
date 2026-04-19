"""
chart.py
Responsible for:
  All chart generation and saving.
  Reads model predictions, outputs PNG files to charts/ folder.

Charts produced:
  chart1_price_indicators.png  - Price + EMA + Bollinger Band (with BB signals marked)
  chart2_lstm_training.png     - LSTM training curve
  chart3_lstm_prediction.png   - LSTM prediction vs actual (BB signal days)
  chart4_arima_prediction.png  - ARIMA prediction vs actual
  chart5_metrics_comparison.png- LSTM vs ARIMA vs Stacking bar chart
  chart6_stacking_result.png   - Stacking detail + Meta Model weights
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from sklearn.metrics import accuracy_score


CHART_DIR = "charts"


def _save(fig, filename: str):
    os.makedirs(CHART_DIR, exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       CHART_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════
#  Chart 1: Price + EMA + Bollinger Band
# ════════════════════════════════════════════════════════
def plot_price_indicators(df, test_start_idx: int,
                          ticker: str, start_date: str, end_date: str):
    fig, ax = plt.subplots(figsize=(16, 5))
    close_all = df["Close"].squeeze()

    ax.plot(df.index, close_all,    "#2196F3", lw=1.2, label="Close")
    ax.plot(df.index, df["EMA_20"], "#FF9800", lw=1, ls="--", label="EMA 20 (Monthly)")
    ax.plot(df.index, df["EMA_60"], "#9C27B0", lw=1, ls="--", label="EMA 60 (Quarterly)")
    ax.fill_between(df.index, df["BB_upper"], df["BB_lower"],
                    alpha=0.08, color="green", label="Bollinger Band")
    ax.plot(df.index, df["BB_upper"], "g-", lw=0.7)
    ax.plot(df.index, df["BB_lower"], "r-", lw=0.7)
    ax.axvline(df.index[test_start_idx], color="gray",
               ls=":", lw=1.5, label="Train/Test Split")

    # Mark BB buy signals (close <= lower band) on price chart
    signal_buy_mask  = df["Signal_buy"] == 1
    signal_sell_mask = df["Signal_sell"] == 1
    if signal_buy_mask.any():
        ax.scatter(df.index[signal_buy_mask],
                   close_all[signal_buy_mask],
                   marker="^", color="#4CAF50", s=60, zorder=5,
                   label="BB Buy Signal (lower band touch)")
    if signal_sell_mask.any():
        ax.scatter(df.index[signal_sell_mask],
                   close_all[signal_sell_mask],
                   marker="v", color="#F44336", s=60, zorder=5,
                   label="BB Sell Signal (upper band touch)")

    ax.set_title(f"[{ticker}] BB Primary Signal  |  ▲Buy=lower touch  ▼Sell=upper touch  ({start_date} ~ {end_date})")
    ax.set_ylabel("Price (TWD)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart1_price_indicators.png")


# ════════════════════════════════════════════════════════
#  Chart 2: LSTM Training Curve
# ════════════════════════════════════════════════════════
def plot_lstm_training(history_log):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(history_log.history["accuracy"],     "#2196F3", lw=1.5, label="Train Accuracy")
    ax.plot(history_log.history["val_accuracy"], "#FF5722", lw=1.5, ls="--", label="Val Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")

    ax_r = ax.twinx()
    ax_r.plot(history_log.history["loss"],     "#4CAF50", alpha=0.6, label="Train Loss")
    ax_r.plot(history_log.history["val_loss"], "#F44336", alpha=0.6, ls="--", label="Val Loss")
    ax_r.set_ylabel("Loss", color="gray")

    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax_r.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=9, loc="upper right")
    ax.set_title("LSTM Training Curve (Accuracy & Loss)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart2_lstm_training.png")


# ════════════════════════════════════════════════════════
#  Chart 3: LSTM Prediction
# ════════════════════════════════════════════════════════
def plot_lstm_prediction(df, test_start_idx, y_te,
                         lstm_preds, lstm_prob,
                         close_actual, close_pred,
                         lstm_acc, lstm_rmse, lstm_mape, ticker):
    test_dates = df.index[test_start_idx: test_start_idx + len(y_te)]
    fig, axes  = plt.subplots(2, 1, figsize=(14, 8))

    # Close price estimation
    idx = range(len(close_actual))
    axes[0].plot(idx, close_actual, "#2196F3", lw=1.5, label="Actual Close")
    axes[0].plot(idx, close_pred,   "#FF5722", lw=1, ls="--",
                 label=f"LSTM Estimate  RMSE={lstm_rmse:.2f}  MAPE={lstm_mape:.2f}%")
    axes[0].fill_between(idx, close_actual, close_pred, alpha=0.12, color="#FF9800")
    axes[0].set_title(f"LSTM Close Estimation  "
                      f"Accuracy={lstm_acc:.2%}  RMSE={lstm_rmse:.2f}  MAPE={lstm_mape:.2f}%")
    axes[0].set_ylabel("Price (TWD)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # Direction prediction
    axes[1].plot(test_dates, y_te,       "#2196F3", lw=1.5, label="Actual",      alpha=0.7)
    axes[1].plot(test_dates, lstm_preds, "#FF5722", lw=1,   label="LSTM Pred",   ls="--")
    axes[1].plot(test_dates, lstm_prob,  "#9E9E9E", lw=0.8, label="LSTM Up Prob")
    axes[1].axhline(0.5, color="gray", lw=0.8, ls=":")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xlabel("Date")
    axes[1].set_title("LSTM Up/Down Prediction")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    fig.suptitle(f"[{ticker}] LSTM Results  (trained on BB buy signal days: close <= BB_lower)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "chart3_lstm_prediction.png")


# ════════════════════════════════════════════════════════
#  Chart 4: ARIMA Prediction
# ════════════════════════════════════════════════════════
def plot_arima_prediction(arima_actual_val, arima_preds_val,
                          arima_preds_dir, arima_acc,
                          arima_rmse, arima_mape, ticker):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    idx = range(len(arima_actual_val))

    # Close price prediction
    axes[0].plot(idx, arima_actual_val, "#2196F3", lw=1.5, label="Actual Close")
    axes[0].plot(idx, arima_preds_val,  "#009688", lw=1, ls="--",
                 label=f"ARIMA Pred  RMSE={arima_rmse:.2f}  MAPE={arima_mape:.2f}%")
    axes[0].fill_between(idx, arima_actual_val, arima_preds_val,
                         alpha=0.12, color="#009688")
    axes[0].set_title(f"ARIMA Close Prediction  "
                      f"Dir.Accuracy={arima_acc:.2%}  RMSE={arima_rmse:.2f}  MAPE={arima_mape:.2f}%")
    axes[0].set_ylabel("Price (TWD)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # Direction
    true_dir = (arima_actual_val[1:] > arima_actual_val[:-1]).astype(int)
    axes[1].plot(range(len(true_dir)), true_dir,             "#2196F3", lw=1.5,
                 label="Actual Dir", alpha=0.7)
    axes[1].plot(range(len(true_dir)), arima_preds_dir[:-1], "#009688", lw=1, ls="--",
                 label="ARIMA Dir")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xlabel("Days")
    axes[1].set_title(f"ARIMA Direction  Accuracy={arima_acc:.2%}")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    fig.suptitle(f"[{ticker}] ARIMA Results", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "chart4_arima_prediction.png")


# ════════════════════════════════════════════════════════
#  Chart 5: Metrics Comparison
# ════════════════════════════════════════════════════════
def plot_metrics_comparison(lstm_acc, arima_acc, stack_acc,
                             lstm_rmse, arima_rmse,
                             lstm_mape, arima_mape, ticker):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"[{ticker}] Model Metrics Comparison  (BB Signal: mean reversion)\n"
        f"Higher Accuracy | Lower RMSE & MAPE is better",
        fontsize=12, fontweight="bold"
    )

    metrics_data = [
        ("Accuracy (%)", lstm_acc*100,  arima_acc*100,  stack_acc*100, "higher"),
        ("RMSE (TWD)",   lstm_rmse,     arima_rmse,     None,          "lower"),
        ("MAPE (%)",     lstm_mape,     arima_mape,     None,          "lower"),
    ]

    for i, (metric, lv, av, sv, direction) in enumerate(metrics_data):
        ax     = axes[i]
        labels = ["LSTM", "ARIMA", "Stacking"] if sv is not None else ["LSTM", "ARIMA"]
        values = [lv, av, sv] if sv is not None else [lv, av]
        colors = ["#FF5722", "#009688", "#9C27B0"][:len(values)]

        bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values)*0.02,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

        better_idx = (np.argmax(values) if direction == "higher"
                      else np.argmin(values))
        bars[better_idx].set_edgecolor("gold")
        bars[better_idx].set_linewidth(3)

        ax.set_title(f"{metric}\n({direction} is better)")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "chart5_metrics_comparison.png")


# ════════════════════════════════════════════════════════
#  Chart 6: Stacking Result
# ════════════════════════════════════════════════════════
def plot_stacking_result(df, test_start_idx, y_te,
                         lstm_preds, arima_preds_dir,
                         stack_pred, stack_prob,
                         lstm_acc, arima_acc, stack_acc,
                         meta_model, ticker):
    # Stacking is computed on aligned signal days only (may be fewer than y_te)
    # Use the minimum length to avoid shape mismatch
    n_common   = min(len(stack_pred), len(y_te), len(lstm_preds))
    test_dates = df.index[test_start_idx: test_start_idx + len(y_te)][:n_common]

    y_te_plot      = y_te[:n_common]
    lstm_plot      = lstm_preds[:n_common]
    arima_plot     = arima_preds_dir[:n_common]
    stack_pred_plt = stack_pred[:n_common]
    stack_prob_plt = stack_prob[:n_common]

    fig, axes  = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(
        f"[{ticker}] Stacking Ensemble  Accuracy={stack_acc:.2%}  (BB Signal days, n={n_common})",
        fontsize=13, fontweight="bold"
    )

    # Model prediction comparison
    ax = axes[0]
    ax.plot(test_dates, y_te_plot,  "#2196F3", lw=2,   label="Actual",                alpha=0.8)
    ax.plot(test_dates, lstm_plot,  "#FF5722", lw=1,   label=f"LSTM  ({lstm_acc:.2%})",  ls="--", alpha=0.7)
    ax.plot(test_dates, arima_plot, "#009688", lw=1,   label=f"ARIMA ({arima_acc:.2%})", ls=":",  alpha=0.7)
    ax.plot(test_dates, stack_pred_plt, "#9C27B0", lw=1.5, label=f"Stack ({stack_acc:.2%})")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Prediction Comparison: LSTM vs ARIMA vs Stacking  (BB signal days only)")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # Stacking probability
    ax = axes[1]
    ax.plot(test_dates, stack_prob_plt, "#9C27B0", lw=1.2, label="Stacking Up Probability")
    ax.fill_between(test_dates, stack_prob_plt, 0.5,
                    where=(stack_prob_plt >= 0.5), alpha=0.2, color="#4CAF50", label="Predicted Up")
    ax.fill_between(test_dates, stack_prob_plt, 0.5,
                    where=(stack_prob_plt <  0.5), alpha=0.2, color="#F44336", label="Predicted Down")
    ax.axhline(0.5, color="gray", lw=1, ls=":")
    ax.set_ylim(0, 1)
    ax.set_title("Stacking Up Probability  (1=reversion UP expected, 0=DOWN)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Meta Model weights
    ax    = axes[2]
    coefs = [meta_model.coef_[0][0], meta_model.coef_[0][1]]
    bars  = ax.bar(["LSTM weight", "ARIMA weight"], coefs,
                   color=["#FF5722", "#009688"], alpha=0.85, width=0.4)
    for bar, val in zip(bars, coefs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(
        f"Meta Model Weights\n"
        f"Positive = bullish signal | Negative = bearish signal\n"
        f"Intercept = {meta_model.intercept_[0]:.4f}"
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "chart6_stacking_result.png")


# ════════════════════════════════════════════════════════
#  Convenience: plot all at once
# ════════════════════════════════════════════════════════
def plot_all(df, test_start_idx, y_te,
             lstm_preds, lstm_prob, arima_preds_dir,
             arima_preds_val, arima_actual_val,
             stack_pred, stack_prob,
             close_actual, close_pred_lstm,
             history_log, meta_model,
             lstm_acc, arima_acc, stack_acc,
             lstm_rmse, arima_rmse,
             lstm_mape, arima_mape,
             ticker, start_date, end_date):

    print(f"\n{'='*55}")
    print(f"  [Chart] Generating all charts")
    print(f"{'='*55}")

    plot_price_indicators(df, test_start_idx, ticker, start_date, end_date)
    plot_lstm_training(history_log)
    plot_lstm_prediction(df, test_start_idx, y_te,
                         lstm_preds, lstm_prob,
                         close_actual, close_pred_lstm,
                         lstm_acc, lstm_rmse, lstm_mape, ticker)
    plot_arima_prediction(arima_actual_val, arima_preds_val,
                          arima_preds_dir, arima_acc,
                          arima_rmse, arima_mape, ticker)
    plot_metrics_comparison(lstm_acc, arima_acc, stack_acc,
                             lstm_rmse, arima_rmse,
                             lstm_mape, arima_mape, ticker)
    plot_stacking_result(df, test_start_idx, y_te,
                         lstm_preds, arima_preds_dir,
                         stack_pred, stack_prob,
                         lstm_acc, arima_acc, stack_acc,
                         meta_model, ticker)

    print(f"\n  All 6 charts saved to ./{CHART_DIR}/")
