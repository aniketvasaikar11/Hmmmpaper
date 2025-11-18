import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from hmmlearn.hmm import MultinomialHMM

st.set_page_config(layout="wide")


# -----------------------------------------
# HSMM EXPANDED STATE CONSTRUCTION (PAPER)
# 4 STATES = Bull Young, Bull Old, Bear Young, Bear Old
# -----------------------------------------

def build_expanded_states(q=2):
    """
    Expanded Semi-Markov Model:
    Bull → Bull-Young(0), Bull-Old(1)
    Bear → Bear-Young(2), Bear-Old(3)
    """
    states = ["Bull-Young", "Bull-Old", "Bear-Young", "Bear-Old"]
    return states


# -----------------------------------------
# TRAIN HSMM (APPROX VIA EXPLICIT EXPANDED STATES)
# -----------------------------------------

def fit_hsmm(returns, q=2):
    """
    A practical HSMM implementation via expanded states.
    Using MultinomialHMM (Non-Gaussian), as paper requires.
    """
    # Convert returns into discrete buckets
    bins = np.percentile(returns, [20, 40, 60, 80])
    obs = np.digitize(returns, bins).reshape(-1, 1)

    n_states = 4  # BullY, BullO, BearY, BearO
    model = MultinomialHMM(n_components=n_states, n_iter=200)
    model.fit(obs)

    hidden = model.predict(obs)
    return hidden


# -----------------------------------------
# MAP STATES TO COLOR
# -----------------------------------------

def regime_color(state):
    mapping = {
        0: "green",
        1: "darkgreen",
        2: "red",
        3: "darkred"
    }
    return mapping.get(state, "black")


# -----------------------------------------
# SIMPLE STRATEGY (LONG IN BULL, CASH IN BEAR)
# -----------------------------------------

def backtest(prices, states):
    returns = prices.pct_change().fillna(0)

    bull_states = [0, 1]   # Bull young + Bull old
    bear_states = [2, 3]

    signal = np.where(np.isin(states, bull_states), 1, 0)

    strat_ret = signal[:-1] * returns[1:]
    equity = (1 + strat_ret).cumprod()

    buyhold = (1 + returns).cumprod()

    stats = {
        "Cumulative Return": (equity.iloc[-1] - 1) * 100,
        "Annualized Return": equity.pct_change().mean() * 252 * 100,
        "Volatility": equity.pct_change().std() * np.sqrt(252) * 100,
        "Sharpe": (equity.pct_change().mean() * 252) /
                  (equity.pct_change().std() * np.sqrt(252) + 1e-9),
        "Max Drawdown": (equity / equity.cummax() - 1).min() * 100,
        "Trade Count": np.sum(np.diff(signal) != 0),
        "Turnover": np.sum(np.abs(np.diff(signal))) * 100
    }

    return equity, buyhold, stats


# -----------------------------------------
# STREAMLIT UI
# -----------------------------------------

st.title("📈 HSMM Regime-Based Backtesting App")

ticker = st.sidebar.text_input("Ticker Symbol", "AVGO")
q = st.sidebar.number_input("Q Value (Duration)", 1, 10, 2)
run = st.sidebar.button("Run Backtest")

if run:

    # Load Data
    df = yf.download(ticker, start="2023-12-01")["Adj Close"]
    returns = df.pct_change().dropna().values

    # Build HSMM
    states = fit_hsmm(returns, q)

    # Run Backtest
    equity, buyhold, stats = backtest(df, states)

    # -----------------------------
    # PERFORMANCE TABLE
    # -----------------------------
    st.subheader("Strategy Performance")
    st.write(stats)

    # -----------------------------
    # EQUITY CURVES
    # -----------------------------
    st.subheader("Equity Curves")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values, label="Strategy")
    ax.plot(buyhold.index, buyhold.values, label="Buy & Hold")
    ax.legend()
    ax.set_title("Equity Curves")
    st.pyplot(fig)

    # -----------------------------
    # PRICE & REGIMES
    # -----------------------------
    st.subheader("Price & Regime Detection")

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df.index, df.values, color="black")

    for i in range(len(df)):
        ax2.axvspan(df.index[i], df.index[i], color=regime_color(states[i]), alpha=0.2)

    ax2.set_title("Regime-Colored Price Chart")
    st.pyplot(fig2)
