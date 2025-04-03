import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from typing import Dict, List, Tuple
import argparse
from backtesting import (
    Backtester,
    Strategy,
    MovingAverageCrossStrategy,
    RSIStrategy,
    MACDStrategy,
    CompositeStrategy,
    parameter_grid_search,
    save_results_to_csv,
)


def generate_sample_data(
    start_date="2018-01-01", end_date="2022-12-31", seed=42
) -> pd.DataFrame:
    """
    샘플 주가 데이터를 생성합니다.

    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
        seed: 난수 시드

    Returns:
        생성된 OHLCV 데이터프레임
    """
    # 날짜 인덱스 생성
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")  # 영업일 기준

    # 난수 시드 설정
    np.random.seed(seed)

    # 초기 가격 설정
    initial_price = 100.0

    # 가격 변동 생성 (랜덤 워크)
    daily_returns = np.random.normal(0.0005, 0.015, len(date_range))
    prices = initial_price * (1 + daily_returns).cumprod()

    # 추세 및 패턴 추가
    t = np.linspace(0, 10, len(date_range))
    trend = 10 * np.sin(t / 5) + 5 * np.sin(t / 2.5)
    prices = prices + trend

    # 조정 기간 추가 (가격 하락)
    correction_start = int(len(date_range) * 0.4)
    correction_end = int(len(date_range) * 0.5)
    correction = np.linspace(0, -20, correction_end - correction_start)
    prices[correction_start:correction_end] += correction

    # 상승 랠리 추가
    rally_start = int(len(date_range) * 0.7)
    rally_end = int(len(date_range) * 0.85)
    rally = np.linspace(0, 30, rally_end - rally_start)
    prices[rally_start:rally_end] += rally

    # 가격이 항상 양수 확보
    prices = np.maximum(prices, 1.0)

    # OHLCV 데이터 생성
    data = pd.DataFrame(index=date_range)
    data["close"] = prices

    # 일일 변동폭 생성
    daily_volatility = prices * np.random.uniform(0.01, 0.03, len(date_range))

    # Open, High, Low 가격 생성
    data["open"] = data["close"].shift(1)
    data["open"].iloc[0] = data["close"].iloc[0] * (1 - np.random.uniform(0, 0.01))

    data["high"] = data[["open", "close"]].max(axis=1) + daily_volatility
    data["low"] = data[["open", "close"]].min(axis=1) - daily_volatility

    # 음수값 방지
    data["low"] = np.maximum(data["low"], 0.1)

    # Volume 생성
    base_volume = np.random.randint(100000, 1000000, len(date_range))
    volume_factor = 1 + np.abs(daily_returns) * 10  # 가격 변동이 클수록 거래량 증가
    data["volume"] = base_volume * volume_factor

    return data


def create_composite_strategy(name: str, params: Dict) -> CompositeStrategy:
    """
    복합 전략을 생성합니다.

    Args:
        name: 전략 이름
        params: 파라미터 딕셔너리

    Returns:
        복합 전략 객체
    """
    # 개별 전략 파라미터 추출
    ma_params = {
        "ma_short": params.get("ma_short", 10),
        "ma_long": params.get("ma_long", 30),
    }

    rsi_params = {
        "rsi_period": params.get("rsi_period", 14),
        "oversold": params.get("oversold", 30),
        "overbought": params.get("overbought", 70),
    }

    macd_params = {
        "macd_fast": params.get("macd_fast", 12),
        "macd_slow": params.get("macd_slow", 26),
        "macd_signal": params.get("macd_signal", 9),
    }

    # 개별 전략 생성
    ma_strategy = MovingAverageCrossStrategy("MA", ma_params)
    rsi_strategy = RSIStrategy("RSI", rsi_params)
    macd_strategy = MACDStrategy("MACD", macd_params)

    # 가중치 설정
    weights = [
        params.get("weight_ma", 1.0),
        params.get("weight_rsi", 1.0),
        params.get("weight_macd", 1.0),
    ]

    # 복합 전략 생성
    composite_params = {
        "threshold": params.get("threshold", 0.5),
        **params,  # 원본 파라미터도 포함
    }

    return CompositeStrategy(
        name=name,
        parameters=composite_params,
        strategies=[ma_strategy, rsi_strategy, macd_strategy],
        weights=weights,
    )


def run_backtest_with_visualization(
    data_path: str = None,
    results_dir: str = "../results",
    charts_dir: str = "../results/charts",
):
    """
    백테스트를 실행하고 결과를 시각화합니다.

    Args:
        data_path: 데이터 파일 경로 (None인 경우 샘플 데이터 생성)
        results_dir: 결과 저장 디렉토리
        charts_dir: 차트 저장 디렉토리
    """
    # 디렉토리 생성
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # 데이터 로드 또는 생성
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        print("Generating sample data")
        data = generate_sample_data()

        # 샘플 데이터 저장
        sample_data_path = os.path.join(results_dir, "sample_data.csv")
        data.to_csv(sample_data_path)
        print(f"Sample data saved to {sample_data_path}")

    # 백테스터 초기화
    initial_capital = 10000.0
    commission = 0.001  # 0.1% 수수료
    backtester = Backtester(data, initial_capital, commission)

    # 1. 개별 전략 백테스트
    print("\n=== Running individual strategy backtests ===")

    # 1.1 Moving Average 전략
    ma_params = {"ma_short": 20, "ma_long": 50}
    ma_strategy = MovingAverageCrossStrategy("MovingAverage", ma_params)
    ma_result = backtester.run(ma_strategy)

    # 1.2 RSI 전략
    rsi_params = {"rsi_period": 14, "oversold": 30, "overbought": 70}
    rsi_strategy = RSIStrategy("RSI", rsi_params)
    rsi_result = backtester.run(rsi_strategy)

    # 1.3 MACD 전략
    macd_params = {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
    macd_strategy = MACDStrategy("MACD", macd_params)
    macd_result = backtester.run(macd_strategy)

    # 2. 복합 전략 백테스트
    print("\n=== Running composite strategy backtest ===")

    composite_params = {
        "ma_short": 20,
        "ma_long": 50,
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "weight_ma": 1.0,
        "weight_rsi": 0.8,
        "weight_macd": 1.2,
        "threshold": 0.5,
    }

    composite_strategy = create_composite_strategy("Composite", composite_params)
    composite_result = backtester.run(composite_strategy)

    # 3. 파라미터 그리드 탐색
    print("\n=== Running parameter grid search ===")

    param_grid = {
        "ma_short": [10, 20, 30],
        "ma_long": [40, 50, 60],
        "rsi_period": [7, 14, 21],
        "weight_ma": [0.8, 1.0, 1.2],
        "weight_rsi": [0.8, 1.0, 1.2],
        "weight_macd": [0.8, 1.0, 1.2],
    }

    all_results, best_result = parameter_grid_search(
        backtester, create_composite_strategy, param_grid, "GridSearch"
    )

    # 4. 결과 저장 및 시각화
    print("\n=== Saving results and generating visualizations ===")

    # 4.1 개별 결과 CSV 저장
    individual_results = [ma_result, rsi_result, macd_result, composite_result]
    individual_df = save_results_to_csv(
        individual_results, os.path.join(results_dir, "individual_results.csv")
    )

    # 4.2 그리드 탐색 결과 CSV 저장
    grid_df = save_results_to_csv(
        all_results, os.path.join(results_dir, "grid_search_results.csv")
    )

    # 4.3 벤치마크 데이터 생성 (Buy & Hold)
    benchmark = pd.Series(
        data["close"] / data["close"].iloc[0] * initial_capital, index=data.index
    )

    # 4.4 개별 전략 시각화
    for result in individual_results:
        # 포트폴리오 성과 차트
        fig = result.plot_portfolio_performance(
            save_path=os.path.join(
                charts_dir, f"{result.strategy_name}_performance.png"
            )
        )
        plt.close(fig)

        # 누적 수익률 차트
        fig = result.plot_cumulative_returns(
            benchmark=benchmark,
            save_path=os.path.join(charts_dir, f"{result.strategy_name}_returns.png"),
        )
        plt.close(fig)

    # 4.5 최적 전략 시각화
    fig = best_result.plot_portfolio_performance(
        save_path=os.path.join(charts_dir, "best_strategy_performance.png")
    )
    plt.close(fig)

    fig = best_result.plot_cumulative_returns(
        benchmark=benchmark,
        save_path=os.path.join(charts_dir, "best_strategy_returns.png"),
    )
    plt.close(fig)

    # 4.6 모든 전략 누적 수익률 비교 차트
    plt.figure(figsize=(12, 8))

    # 각 전략의 누적 수익률
    for result in individual_results:
        portfolio_cum_returns = (result.portfolio_value / initial_capital - 1) * 100
        plt.plot(
            portfolio_cum_returns.index,
            portfolio_cum_returns,
            label=f"{result.strategy_name} ({portfolio_cum_returns.iloc[-1]:.2f}%)",
        )

    # 최적 전략 누적 수익률
    best_cum_returns = (best_result.portfolio_value / initial_capital - 1) * 100
    plt.plot(
        best_cum_returns.index,
        best_cum_returns,
        linewidth=2,
        label=f"Best Strategy ({best_cum_returns.iloc[-1]:.2f}%)",
    )

    # 벤치마크 수익률
    benchmark_cum_returns = (benchmark / initial_capital - 1) * 100
    plt.plot(
        benchmark_cum_returns.index,
        benchmark_cum_returns,
        "k--",
        label=f"Buy & Hold ({benchmark_cum_returns.iloc[-1]:.2f}%)",
    )

    plt.title("Cumulative Returns Comparison")
    plt.ylabel("Cumulative Return (%)")
    plt.xlabel("Date")
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.2)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(
        os.path.join(charts_dir, "strategy_comparison.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("\n=== Backtest completed successfully ===")
    print(f"Results saved to {results_dir}")
    print(f"Charts saved to {charts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trading strategy backtests")
    parser.add_argument(
        "--data", type=str, default=None, help="Path to price data CSV file"
    )
    parser.add_argument(
        "--results", type=str, default="../results", help="Results directory"
    )
    parser.add_argument(
        "--charts", type=str, default="../results/charts", help="Charts directory"
    )

    args = parser.parse_args()

    run_backtest_with_visualization(
        data_path=args.data, results_dir=args.results, charts_dir=args.charts
    )
