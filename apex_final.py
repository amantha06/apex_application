import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def fetch_forex_data(symbol='EURUSD=X', interval='1d', start_date='2010-01-01', end_date=None, save_csv=True, csv_filename='EURUSD_data.csv'):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    forex = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
    
    if forex.empty:
        raise ValueError(f"No data found for symbol {symbol} with interval {interval}.")
    
    forex = forex[['Open', 'High', 'Low', 'Close']]
    forex.ffill(inplace=True)
    forex.bfill(inplace=True)
    
    if save_csv:
        forex.to_csv(csv_filename)
        print(f"Data saved to {csv_filename}")
    
    return forex

def calculate_moving_averages(data, short_window=20, long_window=50):
    data['MA_Short'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['MA_Long'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    return data

def generate_signals(data):
    # Generate signals
    data['Signal'] = 0
    data['Signal'] = np.where(data['MA_Short'] > data['MA_Long'], 1, -1)
    data['Signal'] = data['Signal'].shift(1)  # Shift signals to represent action at next open
    data['Signal'].fillna(0, inplace=True)
    return data

def backtest_strategy(data, initial_capital=100000, position_size_percent=0.05, max_loss_percent=0.10, transaction_cost=10):
    data = data.copy()
    data['Position'] = data['Signal']
    data['Market Returns'] = data['Close'].pct_change()
    data['Capital'] = initial_capital
    data['Position Size'] = 0
    data['PnL'] = 0
    data['Transaction Costs'] = 0
    data['Cumulative PnL'] = 0
    data['Stop Trading'] = False

    for i in range(1, len(data)):
        if data['Stop Trading'].iloc[i-1]:
            data['Position'].iloc[i] = 0
            data['Position Size'].iloc[i] = 0
            data['PnL'].iloc[i] = 0
            data['Capital'].iloc[i] = data['Capital'].iloc[i-1]
            data['Cumulative PnL'].iloc[i] = data['Cumulative PnL'].iloc[i-1]
            data['Stop Trading'].iloc[i] = True
            continue

        # Calculate Position Size based on previous day's capital
        data['Position Size'].iloc[i] = data['Capital'].iloc[i-1] * position_size_percent * data['Position'].iloc[i]

        # Calculate PnL based on previous day's position size and today's market return
        data['PnL'].iloc[i] = data['Position Size'].iloc[i-1] * data['Market Returns'].iloc[i]

        # Calculate transaction costs if position changes
        if data['Position'].iloc[i] != data['Position'].iloc[i-1]:
            data['Transaction Costs'].iloc[i] = transaction_cost

        # Update Capital
        data['Capital'].iloc[i] = data['Capital'].iloc[i-1] + data['PnL'].iloc[i] - data['Transaction Costs'].iloc[i]

        # Update Cumulative PnL
        data['Cumulative PnL'].iloc[i] = data['Capital'].iloc[i] - initial_capital

        # Check for maximum loss threshold
        if data['Capital'].iloc[i] <= initial_capital * (1 - max_loss_percent):
            data['Stop Trading'].iloc[i] = True
            print(f"Trading stopped on {data.index[i].date()} due to maximum loss threshold reached.")
        else:
            data['Stop Trading'].iloc[i] = False

    # Generate trade logs
    trades = data[data['Position'].diff() != 0].copy()
    trades['Action'] = np.where(trades['Position'] > 0, 'Buy', 'Sell')
    trades['Date'] = trades.index
    trades['Price'] = trades['Close']
    trades['PnL'] = trades['PnL']

    # Write trade logs to txt file
    with open('trade_logs.txt', 'w') as f:
        f.write("Date\tAction\tPrice\tPnL\n")
        for _, trade in trades.iterrows():
            f.write(f"{trade['Date'].date()}\t{trade['Action']}\t{trade['Price']:.5f}\t{trade['PnL']:.2f}\n")
    print("Trade logs saved to trade_logs.txt")

    # Calculate performance metrics
    total_pnl = data['Cumulative PnL'].iloc[-1]
    final_portfolio = data['Capital'].iloc[-1]
    total_trades = len(trades)
    winning_trades = trades[trades['PnL'] > 0]
    winning_percentage = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

    # Calculate Sharpe Ratio
    daily_returns = data['PnL'] / data['Capital'].shift(1)
    sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) != 0 else 0

    # Maximum Drawdown
    data['Rolling Max'] = data['Capital'].cummax()
    data['Drawdown'] = (data['Rolling Max'] - data['Capital']) / data['Rolling Max'] * 100
    max_drawdown = data['Drawdown'].max()

    results = {
        'Initial Capital': initial_capital,
        'Final Portfolio': final_portfolio,
        'Total PnL': total_pnl,
        'Total Trades': total_trades,
        'Winning Percentage': winning_percentage,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Data': data
    }

    return results

def plot_equity_curve(data):
    plt.figure(figsize=(14,7))
    plt.plot(data.index, data['Capital'], label='Equity Curve', color='blue')
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Account Balance ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_signals(data):
    plt.figure(figsize=(14,7))
    plt.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
    plt.plot(data.index, data['MA_Short'], label='Short-Term MA', alpha=0.9)
    plt.plot(data.index, data['MA_Long'], label='Long-Term MA', alpha=0.9)

    buy_signals = data[data['Position'].diff() > 0]
    sell_signals = data[data['Position'].diff() < 0]

    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal')

    plt.title('EUR/USD Price with Moving Averages and Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_max_drawdown(data):
    plt.figure(figsize=(14,7))
    plt.plot(data.index, data['Drawdown'], label='Drawdown (%)', color='red')
    plt.title('Drawdown Over Time')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

def generate_performance_summary(results):
    print("\n=== Performance Summary ===")
    print(f"Initial Capital: ${results['Initial Capital']:.2f}")
    print(f"Final Portfolio: ${results['Final Portfolio']:.2f}")
    print(f"Total PnL: ${results['Total PnL']:.2f}")
    print(f"Total Trades: {results['Total Trades']}")
    print(f"Winning Percentage: {results['Winning Percentage']:.2f}%")
    print(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
    print(f"Maximum Drawdown: {results['Max Drawdown']:.2f}%")
    print("============================\n")

if __name__ == "__main__":
    try:
        data = fetch_forex_data(
            symbol='EURUSD=X',
            interval='1d',
            start_date='2015-01-01',
            end_date='2023-10-01',
            save_csv=True,
            csv_filename='EURUSD_daily.csv'
        )
        
        data = calculate_moving_averages(data)
        data = generate_signals(data)
        
        results = backtest_strategy(data)
        generate_performance_summary(results)
        
        plot_equity_curve(results['Data'])
        plot_signals(results['Data'])
        plot_max_drawdown(results['Data'])
        
    except Exception as e:
        print(f"An error occurred: {e}")
