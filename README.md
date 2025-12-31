# Multimodal Stock Forecasting: GCN-LSTM Alpha Agent
### Regime-Aware Geometric Deep Learning for Quantitative Trading

![Equity Curve](Screenshot%202026-01-01%20at%201.57.51%20AM.png)
*> **Figure 1: Walk-Forward Validation Results (15-Months Out-of-Sample).** The Agent (Teal) demonstrates "Defensive Alpha" by decoupling from the market crash in Q4 2024, preserving capital while the benchmark (Grey) drew down -30%.*

---

## 1. Executive Summary
This project implements an institutional-grade trading agent for **NVIDIA (NVDA)** that fuses market structure analysis with non-linear time-series forecasting. Unlike retail models that rely on non-stationary price data, this system utilizes a **Graph Convolutional Network (GCN)** to extract latent sentiment from supply-chain news and fuses it with an **LSTM** network to forecast **Log-Returns** conditional on **Volatility Regimes**.

The strategy was rigorously validated using **Walk-Forward Optimization** (Expanding Window) over 316 trading days, achieving a **Sharpe Ratio of 1.48** and proving robustness across Bull, Bear, and Sideways markets.

---

## 2. Key Quantitative Innovations

### A. Stationary Feature Engineering
* **The Problem:** Deep learning models suffer from "gradient saturation" when asset prices drift significantly (e.g., NVDA rallying from $150 to $1000).
* **The Solution:** Replaced raw OHLCV inputs with **Log-Returns** and **Rolling Volatility** (20-day window). This creates a statistically stationary feature space ($\mu \approx 0, \sigma \approx 1$), allowing the model to generalize patterns across vastly different price levels.

### B. Dynamic Temporal Knowledge Graph
* **The Problem:** Standard sentiment analysis ignores the *structure* of the market (e.g., a TSMC supply shock impacting NVDA).
* **The Solution:** Constructed a daily-evolving Knowledge Graph where:
    * **Nodes:** Market Entities (Competitors, Suppliers, Partners).
    * **Edges:** News-derived sentiment weights.
    * **GCN Layer:** Extracts a spatial "Market Context Vector" ($R^{64}$) for each trading day.
* **No Look-Ahead Bias:** The graph is reconstructed iteratively. On trading day $T$, the graph contains *only* information known at $T-1$.

### C. Robust Validation (Walk-Forward)
* **Methodology:** Abandoned standard Train-Test splits for a realistic simulation of a live trading desk. The model was re-trained every month for 15 months (Jan 2024 - Sept 2025).
* **Result:** Proved model adaptability across changing market regimes (e.g., the "AI Boom" of early 2024 and the "Tech Correction" of late 2024).

---

## 3. Performance Metrics (Out-of-Sample)

| Metric | Value | Institutional Context |
| :--- | :--- | :--- |
| **Sharpe Ratio** | **1.48** | Indicates strong risk-adjusted returns (Target > 1.0). |
| **Total Return** | **>74.3%** | Significantly outperforms the buy-and-hold baseline. |
| **Alpha Generation** | **Confirmed** | Successfully identified "Risk-Off" (Cash) signals during the late-2024 crash. |

---

## 4. Live Inference Engine
The system includes a production-ready dashboard built with **Streamlit** and **Ngrok** for real-time decision support.

![Live Agent Dashboard](Screenshot%202026-01-01%20at%201.25.05%20AM.png)
*> **Figure 2:** The Agent translates complex GNN/LSTM outputs into actionable **Buy / Hold / Cash** signals based on the current volatility regime.*

---

## 5. Tech Stack
* **Core Modeling:** PyTorch, PyTorch Geometric (GNNs), LSTM.
* **Data Engineering:** YFinance, Polygon.io (News), Pandas, NetworkX.
* **Visualization & UI:** Plotly, Streamlit, Matplotlib.
* **Deployment:** Ngrok Tunneling.

---

## 6. Installation & Usage

```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/multimodal-stock-forecasting.git](https://github.com/yourusername/multimodal-stock-forecasting.git)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Live Agent
streamlit run app.py
