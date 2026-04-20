# Stock BB Signal Prediction Model

以布林通道（Bollinger Band）為主要交易訊號，結合 LSTM + ARIMA + Stacking 集成模型，預測每次訊號觸發後持有 N 週的報酬方向（漲或跌）。

---

## 專案架構

```
stock_project/
├── data_loader.py     # 資料下載、特徵計算、BB 訊號生成
├── lstm_model.py      # LSTM 模型定義、訓練、存取
├── arima_model.py     # ARIMA 滾動預測、存取
├── stacking.py        # Stacking 集成 + Meta Model
├── chart.py           # 圖表生成
├── train.py           # 主訓練腳本（多標的 + 週資料）
├── predict.py         # 即時預測腳本
├── models/            # 儲存訓練後的模型
│   ├── lstm_model.h5
│   ├── scalers.pkl         # 各標的獨立 MinMaxScaler
│   ├── arima_multi.pkl     # 各標的 ARIMA 歷史
│   ├── meta_model.pkl      # Stacking Meta Model
│   └── meta.json           # 訓練設定紀錄
└── charts/            # 輸出圖表（PNG）
```

---

## 交易邏輯

### 布林通道主訊號（均值回歸策略）

| 訊號 | 觸發條件 | 預期 |
|------|----------|------|
| **買訊** | 收盤價 ≤ 下軌（超賣） | 均值回歸向上 |
| **賣訊** | 收盤價 ≥ 上軌（超買） | 均值回歸向下 |
| **BB Squeeze** | BB 寬度 < 20 期均值 × 0.85 | 低波動壓縮，突破在即 |

> EMA 20 / EMA 60 仍作為模型特徵輸入，提供趨勢上下文，但不再作為進場門檻。

### 決策流程

```
BB 訊號觸發？
├── 賣訊（close ≥ BB_upper）→ SELL / STAY OUT
├── 買訊（close ≤ BB_lower）
│   ├── Stacking 預測 UP   → BUY（均值回歸可能成功）
│   └── Stacking 預測 DOWN → WAIT（訊號與模型衝突）
├── BB Squeeze（無帶觸碰）  → WATCH（可能突破）
└── 無訊號                  → HOLD / OBSERVE
```

---

## 模型架構

### LSTM（序列分類）
- 輸入：`SEQ_LEN` 週的技術指標序列（19 個特徵）
- 架構：LSTM(64) → LSTM(32) → LSTM(16) → Dense(16) → Sigmoid
- 標籤：買訊觸發後持有 `HOLD_DAYS` 週，`close[t+N] > close[t]` → 1（獲利）

### ARIMA
- 滾動預測下一週收盤價方向
- 各標的獨立訓練，保留最後 200 週歷史

### Stacking Meta Model
- 輸入：`[LSTM 機率, ARIMA 方向]`
- 模型：Logistic Regression
- 訓練：以主 LSTM 在訓練集的 in-sample 預測訓練（無 K-Fold 交叉驗證）

---

## 特徵列表

| 類別 | 特徵 |
|------|------|
| EMA | `EMA_20`, `EMA_60`, `EMA_trend` |
| 布林通道 | `BB_upper`, `BB_mid`, `BB_lower`, `BB_width`, `BB_pct`, `BB_above_upper`, `BB_below_lower`, `BB_squeeze` |
| 成交量 | `VOL_ratio`, `VOL_log` |
| 報酬率 | `Return_1d`, `Return_5d`, `Return_10d` |
| 價格結構 | `High_Low_range` |
| 訊號 | `Signal_buy`, `Signal_sell` |

---

## 訓練設定

| 參數 | 值 | 說明 |
|------|----|------|
| `TICKERS` | 10 檔台股 | 見下方清單 |
| `INTERVAL` | `1wk` | 週線資料 |
| `START_DATE` | `2018-01-01` | 訓練起始日 |
| `SEQ_LEN` | `20` | 20 週歷史上下文（約 5 個月） |
| `HOLD_DAYS` | `4` | 持有 4 週後判定獲利與否 |
| `EPOCHS` | `100` | LSTM 最大訓練輪次（含 EarlyStopping） |
| `BATCH_SIZE` | `32` | 批次大小 |
| `ARIMA_ORDER` | `(2, 1, 2)` | ARIMA(p, d, q) |
| `TRAIN_RATIO` | `0.8` | 80% 訓練 / 20% 測試 |

### 訓練標的

| 代號 | 名稱 |
|------|------|
| 2330.TW | 台積電 |
| 2317.TW | 鴻海 |
| 2454.TW | 聯發科 |
| 3711.TW | 日月光投控 |
| 2383.TW | 台光電 |
| 2382.TW | 廣達 |
| 2303.TW | 聯電 |
| 3017.TW | 奇鋐 |
| 3037.TW | 欣興 |
| 2357.TW | 華碩 |

---

## 執行方式

### 安裝依賴

```bash
pip install yfinance ta tensorflow scikit-learn statsmodels matplotlib
```

### 訓練

```bash
python train.py
```

訓練完成後輸出：
- `models/` — 所有模型檔案
- `charts/` — 7 張分析圖表（含各標的 RMSE / MAPE）

### 預測

```bash
# 使用 meta.json 中第一個標的
python predict.py

# 指定標的（需在訓練清單內）
python predict.py 2330.TW
python predict.py 2454.TW
```

---

## 輸出圖表

| 圖表 | 說明 |
|------|------|
| `chart1_price_indicators.png` | 價格 + EMA + BB，標記買訊（▲）與賣訊（▼）位置 |
| `chart2_lstm_training.png` | LSTM 訓練曲線（Accuracy / Loss） |
| `chart3_lstm_prediction.png` | LSTM 在測試集的預測方向 vs 實際 |
| `chart4_arima_prediction.png` | ARIMA 收盤預測 vs 實際 |
| `chart5_metrics_comparison.png` | LSTM / ARIMA / Stacking 準確率比較 |
| `chart6_stacking_result.png` | Stacking 預測機率 + Meta Model 權重 |
| `chart7_arima_per_ticker.png` | **各標的 ARIMA RMSE、MAPE、方向準確率**（多標的模式） |

---

## 模型儲存格式

| 檔案 | 說明 |
|------|------|
| `lstm_model.h5` | 共用 LSTM 模型（Keras） |
| `scalers.pkl` | `{ticker: MinMaxScaler}` 各標的獨立正規化器 |
| `arima_multi.pkl` | `{ticker: {order, history_tail}}` 各標的 ARIMA 狀態 |
| `meta_model.pkl` | Stacking Logistic Regression |
| `meta.json` | 訓練設定（標的清單、interval、SEQ_LEN、threshold…） |

> 各標的使用獨立 scaler，因各股價格尺度差異大（如台積電 900 元 vs 聯電 50 元）

---

## 注意事項

- 本專案僅供學術研究，不構成任何投資建議
- 訓練樣本數量受限於 BB 訊號觸發頻率（週線約 3–6%），若樣本過少可考慮放寬 `BB_pct < 0.1` 作為觸發條件
- 預測標的若不在訓練清單內，將自動使用最近似標的的 scaler，準確度可能下降
