# A Hybrid GCN-LSTM Ensemble for Structural Alpha Generation

### Project Overview
It is a multi-modal algorithmic trading system designed to predict stock market trends (specifically **NVDA**) by decoding the hidden *structure* of market data.

Unlike traditional time-series models (ARIMA, standard LSTM) that only look at price history, this project constructs **Dynamic Knowledge Graphs** to model the non-Euclidean relationships between price, volatility, and technical indicators. By fusing these spatial graph features with an **LSTM** for temporal reasoning, and stabilizing predictions via a **10-Seed Ensemble**, the model achieves robust "Alpha" over the Buy-and-Hold benchmark.

---

### Key Features
* **Hybrid Architecture:** Combines **Graph Convolutional Networks (GCN)** for spatial feature extraction with **LSTMs** for temporal trend analysis.
* **Dynamic Knowledge Graphs:** Models daily market states as graphs, capturing complex relationships between open, high, low, close, and volume data.
* **Ensemble Optimization:** Utilizes a **10-model voting system** to filter out neural network noise and smooth the equity curve.
* **Directional Loss Function:** Optimized using **Binary Cross Entropy (BCE)** to prioritize directional accuracy (Up/Down) over simple regression, preventing model "flatlining."
* **Strict Backtesting:** Implements a leakage-free evaluation pipeline (2022–2025) with a dedicated out-of-sample test set.

---

### Model Architecture
The system processes data in a three-step pipeline:

1.  **Graph Construction:** Daily market data is converted into a graph where nodes represent features (e.g., RSI, MACD, Price) and edges represent correlations.
2.  **Spatial Learning (GCN):** A Graph Convolutional Network extracts a "Market State Vector" from the daily graph structure.
3.  **Temporal Fusion (LSTM):** A Long Short-Term Memory network takes a sequence of these state vectors (5-day window) to predict the next day's price direction.

---

