"""
arima_model.py
Responsible for:
  1. Rolling ARIMA forecast on test set (for evaluation)
  2. Single-step ARIMA prediction (for predict.py)
  3. Save / Load ARIMA parameters
"""

import os, pickle
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import accuracy_score


# ════════════════════════════════════════════════════════
#  Core: single step prediction
# ════════════════════════════════════════════════════════
def predict_one(history: list, order: tuple):
    """
    Fit ARIMA on history and forecast one step ahead.

    Args:
        history : list of close prices
        order   : (p, d, q) e.g. (2, 1, 2)

    Returns:
        pred : 1=up, 0=down
        yhat : forecasted close price
    """
    try:
        yhat = ARIMA(history, order=order).fit().forecast(steps=1)[0]
    except Exception:
        # Fallback: use last value if ARIMA fails to converge
        yhat = history[-1]
    pred = 1 if yhat > history[-1] else 0
    return pred, float(yhat)


# ════════════════════════════════════════════════════════
#  Rolling forecast on test set
# ════════════════════════════════════════════════════════
def rolling_forecast(close_train,
                     close_test,
                     order: tuple = (2, 1, 2)):
    """
    Walk-forward ARIMA evaluation on test set.
    Each step: fit on all history so far, forecast next day.

    Args:
        close_train : training close prices (Series or list)
        close_test  : test close prices (Series or list)
        order       : ARIMA (p, d, q)

    Returns:
        preds_dir  : array of direction predictions (1=up, 0=down)
        preds_val  : array of forecasted close prices
        actual_val : array of actual close prices
        dir_acc    : direction accuracy
    """
    print(f"\n{'='*55}")
    print(f"  [ARIMA] Rolling forecast  order={order}  n={len(close_test)}")
    print(f"{'='*55}")

    history   = list(close_train)
    preds_dir = []
    preds_val = []

    for i, actual in enumerate(close_test):
        pred_dir, yhat = predict_one(history, order)
        preds_dir.append(pred_dir)
        preds_val.append(yhat)
        history.append(float(actual))

        if (i + 1) % 50 == 0:
            print(f"  Progress {i+1}/{len(close_test)}")

    preds_dir  = np.array(preds_dir)
    preds_val  = np.array(preds_val)
    actual_val = np.array(list(close_test), dtype=float)

    # Direction accuracy: compare consecutive actual prices
    true_dir = (actual_val[1:] > actual_val[:-1]).astype(int)
    dir_acc  = accuracy_score(true_dir, preds_dir[:-1]) if len(true_dir) > 0 else 0.0

    print(f"  Direction Accuracy: {dir_acc:.4f} ({dir_acc:.2%})")

    return preds_dir, preds_val, actual_val, dir_acc


# ════════════════════════════════════════════════════════
#  Predict tomorrow (for predict.py)
# ════════════════════════════════════════════════════════
def predict_tomorrow(arima_meta: dict,
                     new_close_series,
                     history_limit: int = 200):
    """
    Predict next day direction using saved ARIMA history.

    Args:
        arima_meta       : dict with 'order' and 'history_tail'
        new_close_series : latest close prices to append
        history_limit    : max history length to keep (avoid slow computation)

    Returns:
        pred : 1=up, 0=down
        yhat : forecasted close price
    """
    order   = tuple(arima_meta["order"])
    history = list(arima_meta["history_tail"])
    last    = history[-1]

    # Append new data points not already in history
    for c in new_close_series:
        if c != last:
            history.append(float(c))

    # Limit history length for speed
    history = history[-history_limit:]

    pred, yhat = predict_one(history, order)
    return pred, yhat


# ════════════════════════════════════════════════════════
#  Save / Load
# ════════════════════════════════════════════════════════
def save(order: tuple, close_series, model_dir: str = "models"):
    """Save single-ticker ARIMA params (legacy)."""
    os.makedirs(model_dir, exist_ok=True)
    arima_meta = {
        "order"        : list(order),
        "history_tail" : list(close_series),
    }
    path = os.path.join(model_dir, "arima_params.pkl")
    with open(path, "wb") as f:
        pickle.dump(arima_meta, f)
    print(f"  ARIMA params -> {path}")


def save_multi(arima_dict: dict, model_dir: str = "models"):
    """
    Save per-ticker ARIMA params dict.

    Args:
        arima_dict : {ticker: {"order": [...], "history_tail": [...]}}
        model_dir  : output directory
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "arima_multi.pkl")
    with open(path, "wb") as f:
        pickle.dump(arima_dict, f)
    print(f"  ARIMA multi  -> {path}  ({len(arima_dict)} tickers)")


def load(model_dir: str = "models") -> dict:
    """Load single-ticker ARIMA params."""
    path = os.path.join(model_dir, "arima_params.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_multi(model_dir: str = "models") -> dict:
    """Load per-ticker ARIMA params dict saved by save_multi()."""
    path = os.path.join(model_dir, "arima_multi.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path} — run train.py with multi-ticker mode first.")
    with open(path, "rb") as f:
        return pickle.load(f)
