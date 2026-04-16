# Quantitative Backtesting Framework — CATL Case Study

A Python backtesting framework comparing three classic quantitative
trading strategies on Contemporary Amperex Technology Co. Limited
(CATL, 300750.SZ), the dominant player in China's EV battery industry.

## Strategies Compared
- Moving Average Crossover (20/60)
- Momentum (20-day lookback, 5% threshold)
- Mean Reversion (20-day Bollinger Band, 2σ)

## Key Findings
On a strongly-trending single stock (Jan 2020 – Apr 2026):
- Buy-and-hold returned 838% but with ~80% maximum drawdown
- Momentum achieved the best risk-adjusted return (Sharpe 0.94)
  with only 28% drawdown
- Mean Reversion underperformed dramatically in a trending market

See `research_report.pdf` for full methodology and discussion.

## Usage
```bash
pip install akshare pandas numpy matplotlib
python backtest.py
```

## Author
 LIU, Jiachen Kevin, April 2026
