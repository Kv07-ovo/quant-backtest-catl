"""
宁德时代(300750) 三策略回测对比
====================================
比较三种经典量化策略在同一标的上的历史表现:
  1. 双均线策略 (MA Crossover)   — 趋势跟踪代表
  2. 动量策略   (Momentum)        — 追涨代表
  3. 均值回归   (Mean Reversion)  — 抄底代表

运行前请先安装依赖:
    pip install akshare pandas numpy matplotlib

运行:
    python backtest.py

输出:
    - 控制台打印三策略绩效对比表
    - backtest_result.png  净值曲线图
    - trades.csv           所有交易记录
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  配置区 —— 想改参数就改这里
# ============================================================
STOCK_CODE = "300750"        # 股票代码:宁德时代
START_DATE = "20200101"      # 回测起始日
END_DATE   = "20260415"      # 回测结束日
INITIAL_CAPITAL = 100_000    # 初始资金 10 万
COMMISSION = 0.0003          # 券商佣金 0.03% (双边)
STAMP_TAX  = 0.001           # 印花税 0.1%  (仅卖出)


# ============================================================
#  1. 数据获取
# ============================================================
def fetch_data(code: str, start: str, end: str) -> pd.DataFrame:
    """从 akshare 拉取 A 股日 K 线(前复权)"""
    df = ak.stock_zh_a_hist(
        symbol=code, period="daily",
        start_date=start, end_date=end,
        adjust="qfq",
    )
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "close", "high", "low", "volume"]]


# ============================================================
#  2. 三种策略 —— 统一返回 signal 序列
#     signal = 1 表示应持仓, 0 表示应空仓
# ============================================================
def strategy_ma_crossover(df: pd.DataFrame, short: int = 20, long: int = 60) -> pd.Series:
    """双均线: 短均线 > 长均线时持有"""
    ma_short = df["close"].rolling(short).mean()
    ma_long  = df["close"].rolling(long).mean()
    return (ma_short > ma_long).astype(int)


def strategy_momentum(df: pd.DataFrame, lookback: int = 20, threshold: float = 0.05) -> pd.Series:
    """动量: 过去 N 日收益 > 阈值时持有"""
    ret_n = df["close"].pct_change(lookback)
    return (ret_n > threshold).astype(int)


def strategy_mean_reversion(df: pd.DataFrame, window: int = 20, k: float = 2.0) -> pd.Series:
    """
    均值回归:
      价格跌破 MA - k*σ (布林下轨) 时买入持有,
      价格回升至 MA 以上时卖出.
    """
    ma  = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    lower = ma - k * std

    signal = pd.Series(0, index=df.index, dtype=int)
    holding = False
    for i in range(len(df)):
        if np.isnan(ma.iloc[i]):
            continue
        price = df["close"].iloc[i]
        if not holding and price < lower.iloc[i]:
            holding = True
        elif holding and price > ma.iloc[i]:
            holding = False
        signal.iloc[i] = 1 if holding else 0
    return signal


# ============================================================
#  3. 回测引擎
#     约定: T 日收盘产生信号, T+1 日开盘执行, 简化为用 T+1 收盘价成交
# ============================================================
def backtest(df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    s = df.copy()
    s["signal"]   = signal
    s["position"] = s["signal"].shift(1).fillna(0)        # T+1 执行
    s["daily_ret"]    = s["close"].pct_change().fillna(0)
    s["strategy_ret"] = s["position"] * s["daily_ret"]

    # 换仓成本: 买入收佣金, 卖出收佣金+印花税
    pos_change = s["position"].diff().fillna(0)
    buy_cost  = (pos_change > 0) * COMMISSION
    sell_cost = (pos_change < 0) * (COMMISSION + STAMP_TAX)
    s["cost"] = buy_cost + sell_cost
    s["strategy_ret"] = s["strategy_ret"] - s["cost"]

    # 净值曲线
    s["equity"]    = (1 + s["strategy_ret"]).cumprod() * INITIAL_CAPITAL
    s["benchmark"] = (1 + s["daily_ret"]).cumprod() * INITIAL_CAPITAL
    return s


# ============================================================
#  4. 绩效指标
# ============================================================
def calc_metrics(s: pd.DataFrame, name: str) -> dict:
    equity = s["equity"]
    returns = s["strategy_ret"]

    total_return = equity.iloc[-1] / INITIAL_CAPITAL - 1
    n_years = len(s) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    drawdown = (equity - equity.cummax()) / equity.cummax()
    max_dd = drawdown.min()

    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    n_trades = int((s["position"].diff().fillna(0) != 0).sum() / 2)

    return {
        "策略": name,
        "总收益率": f"{total_return:.2%}",
        "年化收益": f"{annual_return:.2%}",
        "最大回撤": f"{max_dd:.2%}",
        "夏普比率": f"{sharpe:.2f}",
        "交易次数": n_trades,
    }


# ============================================================
#  5. 可视化 & 导出
# ============================================================
def plot_comparison(results: dict, save_path: str = "backtest_result.png"):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "PingFang SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for (name, s), color in zip(results.items(), colors):
        ax.plot(s.index, s["equity"], label=name, linewidth=1.6, color=color)

    first = next(iter(results.values()))
    ax.plot(first.index, first["benchmark"],
            label="基准(买入持有)", linestyle="--", color="gray", linewidth=1.5)

    ax.set_title(f"{STOCK_CODE} 三策略回测对比  ({START_DATE}→{END_DATE})", fontsize=14)
    ax.set_xlabel("日期"); ax.set_ylabel("账户净值 (元)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()


def export_trades(results: dict, save_path: str = "trades.csv"):
    rows = []
    for name, s in results.items():
        chg = s["position"].diff().fillna(0)
        trade_days = s[chg != 0]
        for dt, row in trade_days.iterrows():
            rows.append({
                "日期": dt.date(),
                "策略": name,
                "动作": "买入" if row["position"] == 1 else "卖出",
                "价格": round(row["close"], 2),
                "当日净值": round(row["equity"], 2),
            })
    if rows:
        pd.DataFrame(rows).to_csv(save_path, encoding="utf-8-sig", index=False)


# ============================================================
#  6. 主流程
# ============================================================
def main():
    print(f"拉取 {STOCK_CODE} 数据 ({START_DATE} → {END_DATE}) ...")
    df = fetch_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"共 {len(df)} 个交易日  {df.index[0].date()} → {df.index[-1].date()}\n")

    strategies = {
        "双均线 20/60":    strategy_ma_crossover(df, 20, 60),
        "动量 20日>5%":    strategy_momentum(df, 20, 0.05),
        "均值回归 20/2σ":  strategy_mean_reversion(df, 20, 2.0),
    }

    results, metrics = {}, []
    for name, sig in strategies.items():
        res = backtest(df, sig)
        results[name] = res
        metrics.append(calc_metrics(res, name))

    # 加入基准做对比
    bench_return = results[next(iter(results))]["benchmark"].iloc[-1] / INITIAL_CAPITAL - 1
    metrics.append({
        "策略": "基准(买入持有)",
        "总收益率": f"{bench_return:.2%}",
        "年化收益": f"{(1 + bench_return) ** (252 / len(df)) - 1:.2%}",
        "最大回撤": "—", "夏普比率": "—", "交易次数": 1,
    })

    print("=" * 76)
    print("回测绩效对比")
    print("=" * 76)
    print(pd.DataFrame(metrics).to_string(index=False))
    print("=" * 76)

    plot_comparison(results)
    export_trades(results)
    print("\n✓ 图表已保存: backtest_result.png")
    print("✓ 交易记录已保存: trades.csv")


if __name__ == "__main__":
    main()
