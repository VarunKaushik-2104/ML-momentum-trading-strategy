import pandas as pd
import numpy as np
import os
# Load Data
data = pd.read_csv("data/processed_data.csv")
preds = pd.read_csv("outputs/advanced_predictions.csv")

data = data.dropna(subset=['Date'])
preds = preds.dropna(subset=['Date'])

# Merge Returns with Predictions
df = preds.merge(
    data[['Date', 'Stock', 'Forward_Return_5']],
    on=['Date', 'Stock'],
    how='left'
)

# Portfolio Construction
TOP_N = 10
portfolio_returns = []

for date, group in df.groupby('Date'):
    top_stocks = group.sort_values(
        'Predicted_Prob', ascending=False
    ).head(TOP_N)

    daily_return = top_stocks['Forward_Return_5'].mean()
    portfolio_returns.append([date, daily_return])

portfolio = pd.DataFrame(
    portfolio_returns,
    columns=['Date', 'Portfolio_Return']
)

# Performance Metrics
portfolio['Cumulative_Return'] = (1 + portfolio['Portfolio_Return']).cumprod()

cagr = portfolio['Cumulative_Return'].iloc[-1] ** (252 / len(portfolio)) - 1
sharpe = (
    portfolio['Portfolio_Return'].mean() /
    portfolio['Portfolio_Return'].std()
) * np.sqrt(252)

max_dd = (
    portfolio['Cumulative_Return'] /
    portfolio['Cumulative_Return'].cummax() - 1
).min()

# Save Results
os.makedirs("outputs", exist_ok=True)

portfolio.to_csv("outputs/final_portfolio_returns.csv", index=False)

metrics = pd.DataFrame({
    'Metric': ['CAGR', 'Sharpe Ratio', 'Max Drawdown'],
    'Value': [cagr, sharpe, max_dd]
})

metrics.to_csv("outputs/final_metrics.csv", index=False)
print(metrics)
