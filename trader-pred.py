#!/usr/bin/env python3

import argparse
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Backtester engine (now supports per-trade fee)
# ------------------------------------------------------------------
def backtest(df, signals, fee=0.0):
    cash = 1000.0
    pos = 0.0
    equity = []
    trades = []

    for date, signal in signals.items():
        price = df.at[date, 'close']
        if signal == 1 and cash > 0.0:
            # BUY: invest cash after fee
            invest_amt = cash * (1.0 - fee)
            pos = invest_amt / price
            cash = 0.0
            trades.append((date, price, 'buy'))
        elif signal == -1 and pos > 0.0:
            # SELL: liquidate position minus fee
            proceeds = pos * price
            cash = proceeds * (1.0 - fee)
            pos = 0.0
            trades.append((date, price, 'sell'))

        equity.append(cash + pos * price)

    eqs = pd.Series(equity, index=signals.index)
    daily_ret = eqs.pct_change().fillna(0)
    total_ret = eqs.iloc[-1] / 1000.0 - 1
    ann_ret = (1 + daily_ret.mean())**252 - 1
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else np.nan)
    max_dd = ((eqs - eqs.cummax()) / eqs.cummax()).min()

    return {
        'Total Return': total_ret,
        'Ann. Return': ann_ret,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Num Trades': len(trades)
    }, eqs, trades

# ------------------------------------------------------------------
# Strategy definitions
# ------------------------------------------------------------------
def strat_buy_hold(df, horizon=0, change=0.0):
    sig = pd.Series(0, index=df.index)
    sig.iloc[0] = 1
    sig.iloc[-1] = -1
    return sig

def strat_predicted(df, horizon=0, change=0.0):
    if 'close_predicted_1' not in df.columns:
        raise KeyError('DataFrame must contain close_predicted_1 column for strat_predicted')

    preds = df['close_predicted_1']
    sig = pd.Series(0, index=df.index)

    if horizon < 1:
        # compare actual close vs. predicted, with threshold 'change'
        pct_diff = (preds.shift(1) - df['close']) / df['close']
        sig[pct_diff >  change] = 1
        sig[pct_diff < -change] = -1
    else:
        # compare predicted[t] vs. predicted[t - horizon], with threshold
        for i in range(horizon, len(preds) - 1):
            prev = preds.iloc[i - horizon + 1]
            curr = preds.iloc[i + 1]
            rel_change = (curr - prev) / prev if prev != 0 else 0.0
            if rel_change >  change:
                sig.iloc[i] = 1
            elif rel_change < -change:
                sig.iloc[i] = -1

    return sig

# ------------------------------------------------------------------
# Main program
# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description='Backtest strategies with optional per-trade fee and signal threshold'
    )
    p.add_argument('filename',
                   help='CSV file (date + OHLC + optional close_predicted_1)')
    p.add_argument(
        '--horizon', '-n',
        type=int,
        default=0,
        help='bars ahead that close_predicted_1 refers to (default: 0)'
    )
    p.add_argument(
        '--fee', '-f',
        type=float,
        default=0.0,
        help='per-trade fee as fraction (e.g. 0.01 = 1%%)'
    )
    p.add_argument(
        '--change', '-c',
        type=float,
        default=0.0,
        help='minimum percent change threshold for signals (e.g. 0.02 = 2%%)'
    )
    args = p.parse_args()

    # --- load CSV ---
    try:
        df = pd.read_csv(args.filename, header=0)
    except Exception as e:
        print(f'❌ Error reading {args.filename}: {e}', file=sys.stderr)
        sys.exit(1)

    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
    if not date_col:
        print('❌ No date/time column found.', file=sys.stderr)
        sys.exit(1)

    # parse dates (epoch or ISO strings)
    try:
        col = df[date_col]
        if pd.api.types.is_numeric_dtype(col):
            df['dt'] = pd.to_datetime(col, unit='s', errors='raise')
        else:
            df['dt'] = pd.to_datetime(col, errors='raise')
    except Exception as e:
        print(f'❌ Could not parse {date_col}: {e}', file=sys.stderr)
        sys.exit(1)

    df.set_index('dt', inplace=True)
    for col in ('open', 'high', 'low', 'close'):
        if col not in df.columns:
            print(f'❌ Missing required column: {col}', file=sys.stderr)
            sys.exit(1)

    # handle predictions & compute shifted timestamps for plotting
    if 'close_predicted_1' in df.columns:
        df['close_predicted_1'].fillna(df['close'], inplace=True)
        h = args.horizon
        if h != 0:
            deltas = df.index.to_series().diff().dropna()
            avg_delta = deltas.mean()
            df['pred_dt'] = df.index + avg_delta * h
        else:
            df['pred_dt'] = df.index

    # choose strategies
    strategies = {
#        'BuyHold': strat_buy_hold,
        'Predicted': strat_predicted
    }

    results = []
    curves = {}
    trades_map = {}

    for name, func in strategies.items():
        sig, = [None]  # placeholder to satisfy unpacking below
        sig = func(df, args.horizon, args.change)
        metrics, eq, trades = backtest(df, sig, args.fee)
        metrics['Strategy'] = name
        results.append(metrics)
        curves[name] = eq
        trades_map[name] = trades

    # print performance summary
    summary = pd.DataFrame(results).set_index('Strategy')
    print('\nStrategy Performance Summary:\n')
    print(summary.to_string(float_format='{:.4f}'.format))

    # plot best
    best = summary['Total Return'].idxmax()
    price = df['close']
    eq = curves[best]
    trades = trades_map[best]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price.index, y=price.values,
        name='Actual Close', yaxis='y1', mode='lines'
    ))

    if 'close_predicted_1' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['pred_dt'], y=df['close_predicted_1'],
            name='Predicted Close', yaxis='y1',
            mode='lines', line=dict(dash='dash')
        ))

    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        name='Portfolio Value', yaxis='y2', mode='lines'
    ))

    buys  = [(d, p) for d, p, s in trades if s == 'buy']
    sells = [(d, p) for d, p, s in trades if s == 'sell']

    if buys:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in buys],
            y=[p for _, p in buys],
            mode='markers', name='Buy',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))
    if sells:
        fig.add_trace(go.Scatter(
            x=[d for d, _ in sells],
            y=[p for _, p in sells],
            mode='markers', name='Sell',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))

    fig.update_layout(
        title=f'Price & Equity — Best Strategy: {best}',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price', side='left'),
        yaxis2=dict(title='Portfolio Value ($)', overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=60, r=60, t=50, b=50)
    )

    fig.show()

if __name__ == '__main__':
    main()