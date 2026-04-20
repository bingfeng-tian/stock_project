"""
chart.py — one function = one figure = one PNG file

Chart index:
  chart01_confusion_matrix.png      Confusion matrix  (LSTM vs Stacking)
  chart02_roc_curve.png             ROC curve + AUC   (LSTM vs Stacking)
  chart03_prob_distribution.png     Predicted probability distribution by actual label
  chart04_threshold_sensitivity.png Accuracy / Precision / Recall vs classification threshold
  chart05_arima_rmse_per_ticker.png ARIMA RMSE for each ticker
  chart06_arima_mape_per_ticker.png ARIMA MAPE for each ticker
  chart07_arima_acc_per_ticker.png  ARIMA direction accuracy for each ticker
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, roc_auc_score,
    precision_score, recall_score, accuracy_score,
)

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

CHART_DIR = "charts"


# ────────────────────────────────────────────────────────
#  Internal helpers
# ────────────────────────────────────────────────────────
def _save(fig, filename: str):
    os.makedirs(CHART_DIR, exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       CHART_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved -> {out}")
    plt.close(fig)


def _draw_cm(ax, cm, title, acc):
    """Render a single 2×2 confusion matrix onto ax."""
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{title}\nAccuracy = {acc:.2%}", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Down(0)", "Pred: Up(1)"])
    ax.set_yticklabels(["Actual: Down(0)", "Actual: Up(1)"])
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")


# ════════════════════════════════════════════════════════
#  chart01 — Confusion Matrix  (LSTM | Stacking)
# ════════════════════════════════════════════════════════
def plot_confusion_matrix(y_te, lstm_pred, stack_pred,
                          lstm_acc: float, stack_acc: float,
                          ticker: str = "Multi-Ticker"):
    n = min(len(y_te), len(lstm_pred), len(stack_pred))
    y  = np.array(y_te[:n],     dtype=int)
    lp = np.array(lstm_pred[:n], dtype=int)
    sp = np.array(stack_pred[:n], dtype=int)

    cm_lstm  = confusion_matrix(y, lp,  labels=[0, 1])
    cm_stack = confusion_matrix(y, sp,  labels=[0, 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    _draw_cm(ax1, cm_lstm,  f"LSTM  [{ticker}]",     lstm_acc)
    _draw_cm(ax2, cm_stack, f"Stacking  [{ticker}]", stack_acc)
    fig.suptitle(
        "Confusion Matrix  —  BB buy signal days  (will it be profitable?)",
        fontsize=13, y=1.02
    )
    fig.tight_layout()
    _save(fig, "chart01_confusion_matrix.png")


# ════════════════════════════════════════════════════════
#  chart02 — ROC Curve + AUC  (LSTM | Stacking)
# ════════════════════════════════════════════════════════
def plot_roc_curve(y_te, lstm_prob, stack_prob,
                   ticker: str = "Multi-Ticker"):
    fig, ax = plt.subplots(figsize=(7, 6))

    for prob, label, color in [
        (lstm_prob,  "LSTM",     "#FF5722"),
        (stack_prob, "Stacking", "#9C27B0"),
    ]:
        if len(prob) == 0:
            continue
        n = min(len(y_te), len(prob))
        try:
            fpr, tpr, _ = roc_curve(y_te[:n], prob[:n])
            auc = roc_auc_score(y_te[:n], prob[:n])
            ax.plot(fpr, tpr, lw=2.5,
                    label=f"{label}  AUC = {auc:.3f}", color=color)
        except Exception as e:
            print(f"  ROC skip ({label}): {e}")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random  AUC = 0.500")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(
        f"[{ticker}]  ROC Curve + AUC\n"
        f"(BB buy signal days: profitable prediction)"
    )
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, "chart02_roc_curve.png")


# ════════════════════════════════════════════════════════
#  chart03 — Predicted Probability Distribution
# ════════════════════════════════════════════════════════
def plot_prob_distribution(y_te, stack_prob, stack_acc: float,
                           ticker: str = "Multi-Ticker"):
    n  = min(len(y_te), len(stack_prob))
    y  = np.array(y_te[:n],      dtype=int)
    sp = np.array(stack_prob[:n], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 21)

    ax.hist(sp[y == 1], bins=bins, alpha=0.68, color="#4CAF50",
            label=f"Actual UP  (n={int((y==1).sum())})", edgecolor="white")
    ax.hist(sp[y == 0], bins=bins, alpha=0.68, color="#F44336",
            label=f"Actual DOWN  (n={int((y==0).sum())})", edgecolor="white")
    ax.axvline(0.5, color="gray",  lw=1.5, ls="--", label="threshold = 0.5")

    ax.set_xlabel("Stacking Up-Probability")
    ax.set_ylabel("Count")
    ax.set_title(
        f"[{ticker}]  Predicted Probability Distribution\n"
        f"Accuracy = {stack_acc:.2%}  |  "
        f"Good separation → stronger discriminative power"
    )
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart03_prob_distribution.png")


# ════════════════════════════════════════════════════════
#  chart04 — Threshold Sensitivity
# ════════════════════════════════════════════════════════
def plot_threshold_sensitivity(y_te, stack_prob,
                               ticker: str = "Multi-Ticker"):
    n  = min(len(y_te), len(stack_prob))
    y  = np.array(y_te[:n],      dtype=int)
    sp = np.array(stack_prob[:n], dtype=float)

    thresholds = np.arange(0.30, 0.71, 0.02)
    accs, precs, recs, coverages = [], [], [], []

    for thr in thresholds:
        pred     = (sp >= thr).astype(int)
        coverage = pred.sum() / len(pred) if len(pred) > 0 else 0.0
        coverages.append(coverage)
        if pred.sum() == 0:
            accs.append(np.nan); precs.append(np.nan); recs.append(np.nan)
        else:
            accs.append(accuracy_score(y, pred))
            precs.append(precision_score(y, pred, zero_division=0))
            recs.append(recall_score(y, pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(thresholds, accs,  "#2196F3", lw=2.5, marker="o", ms=5,
            label="Accuracy")
    ax.plot(thresholds, precs, "#4CAF50", lw=2.5, marker="s", ms=5,
            label="Precision")
    ax.plot(thresholds, recs,  "#FF5722", lw=2.5, marker="^", ms=5,
            label="Recall")

    ax2 = ax.twinx()
    ax2.bar(thresholds, coverages, width=0.015, alpha=0.18,
            color="gray", label="Signal coverage")
    ax2.set_ylabel("Signal Coverage (fraction of test signals entered)",
                   color="gray", fontsize=9)
    ax2.set_ylim(0, 2.0)

    ax.axhline(0.5,  color="gray", lw=0.8, ls=":")
    ax.axvline(0.5,  color="gold", lw=1.5, ls="--", label="default thr = 0.5")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score (0 – 1)")
    ax.set_ylim(0, 1.08)
    ax.set_title(
        f"[{ticker}]  Threshold Sensitivity\n"
        f"Higher threshold → fewer but more confident BUY signals"
    )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart04_threshold_sensitivity.png")


# ════════════════════════════════════════════════════════
#  chart05 — ARIMA RMSE per Ticker
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
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    best = int(np.argmin(rmse_vals))
    bars[best].set_edgecolor("gold"); bars[best].set_linewidth(2.5)
    ax.axhline(avg, color="red", lw=1.2, ls="--", label=f"avg = {avg:.1f}")
    ax.set_xlabel("Ticker"); ax.set_ylabel("RMSE (TWD)")
    ax.set_title("ARIMA Close Price RMSE per Ticker  ↓ lower is better")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart05_arima_rmse_per_ticker.png")


# ════════════════════════════════════════════════════════
#  chart06 — ARIMA MAPE per Ticker
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
                f"{val:.2f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    best = int(np.argmin(mape_vals))
    bars[best].set_edgecolor("gold"); bars[best].set_linewidth(2.5)
    ax.axhline(avg, color="red", lw=1.2, ls="--", label=f"avg = {avg:.2f}%")
    ax.set_xlabel("Ticker"); ax.set_ylabel("MAPE (%)")
    ax.set_title("ARIMA Close Price MAPE per Ticker  ↓ lower is better")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart06_arima_mape_per_ticker.png")


# ════════════════════════════════════════════════════════
#  chart07 — ARIMA Direction Accuracy per Ticker
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
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    best = int(np.argmax(acc_vals))
    bars[best].set_edgecolor("gold"); bars[best].set_linewidth(2.5)
    ax.axhline(50,  color="gray", lw=1,   ls=":",  label="random baseline 50%")
    ax.axhline(avg, color="red",  lw=1.2, ls="--", label=f"avg = {avg:.1f}%")
    ax.set_xlabel("Ticker"); ax.set_ylabel("Direction Accuracy (%)")
    ax.set_title("ARIMA Direction Accuracy per Ticker  ↑ higher is better")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart07_arima_acc_per_ticker.png")


# ════════════════════════════════════════════════════════
#  chart08 — LSTM Training Curve  (accuracy + loss, twin axis)
# ════════════════════════════════════════════════════════
def plot_lstm_training(history_log, ticker: str = "Multi-Ticker"):
    hist = history_log.history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Accuracy on left axis
    ax.plot(epochs, hist["accuracy"],     "#2196F3", lw=2,   label="Train Accuracy")
    ax.plot(epochs, hist["val_accuracy"], "#FF5722", lw=2,   label="Val Accuracy", ls="--")
    ax.set_ylabel("Accuracy"); ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05)

    # Loss on right axis
    ax2 = ax.twinx()
    ax2.plot(epochs, hist["loss"],     "#4CAF50", lw=1.5, alpha=0.7, label="Train Loss")
    ax2.plot(epochs, hist["val_loss"], "#F44336", lw=1.5, alpha=0.7, label="Val Loss", ls="--")
    ax2.set_ylabel("Loss", color="gray")

    best_epoch = int(np.argmin(hist["val_loss"])) + 1
    ax.axvline(best_epoch, color="gold", lw=1.5, ls=":",
               label=f"Best epoch = {best_epoch}")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")
    ax.set_title(
        f"[{ticker}]  LSTM Training Curve\n"
        f"Best val_loss at epoch {best_epoch}  "
        f"(val_acc = {hist['val_accuracy'][best_epoch-1]:.2%})"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "chart08_lstm_training.png")


# ════════════════════════════════════════════════════════
#  Convenience: called from train.py
# ════════════════════════════════════════════════════════
def plot_arima_metrics_by_ticker(ticker_stats: list):
    """Generate chart05, chart06, chart07 in one call."""
    print(f"\n{'='*55}")
    print(f"  [Chart] Generating per-ticker metric charts")
    print(f"{'='*55}")
    plot_arima_rmse_per_ticker(ticker_stats)
    plot_arima_mape_per_ticker(ticker_stats)
    plot_arima_acc_per_ticker(ticker_stats)
    print(f"  Charts 05–07 saved to ./{CHART_DIR}/")
