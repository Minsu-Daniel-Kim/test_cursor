import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from typing import Dict, List, Tuple
import argparse
import time
import warnings

# 경고 메시지 필터링
warnings.filterwarnings("ignore")

from backtesting import (
    Backtester,
    Strategy,
    MovingAverageCrossStrategy,
    RSIStrategy,
    MACDStrategy,
    CompositeStrategy,
    parameter_grid_search,
    save_results_to_csv,
    BacktestResult,
)
from ml_strategies import LightGBMStrategy
from run_backtest import generate_sample_data, create_composite_strategy


def run_lightgbm_strategy(
    data: pd.DataFrame, initial_capital: float = 10000.0, commission: float = 0.001
) -> BacktestResult:
    """
    LightGBM 기반 전략 백테스트 실행

    Args:
        data: OHLCV 데이터
        initial_capital: 초기 자본금
        commission: 거래 수수료

    Returns:
        백테스트 결과
    """
    print("\n=== Running LightGBM strategy backtest ===")

    # LightGBM 전략 파라미터 설정
    lgb_params = {
        "prediction_horizon": 5,  # 5일 후 수익률 예측
        "train_interval": 60,  # 60일마다 모델 재학습
        "features_lookback": 20,  # 특성 계산을 위한 룩백 기간
        "num_boost_round": 100,  # 부스팅 라운드 수
        "threshold_buy": 0.01,  # 1% 이상 상승 예상 시 매수
        "threshold_sell": -0.01,  # 1% 이상 하락 예상 시 매도
        "lgb_params": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        },
    }

    # LightGBM 전략 생성
    lgb_strategy = LightGBMStrategy("LightGBM", lgb_params)

    # 백테스터 초기화
    backtester = Backtester(data, initial_capital, commission)

    # 백테스트 실행
    start_time = time.time()
    lgb_result = backtester.run(lgb_strategy)
    elapsed = time.time() - start_time

    print(f"LightGBM backtest completed in {elapsed:.2f}s")
    print(f"Total Return: {lgb_result.total_return:.2f}%")
    print(f"Sharpe Ratio: {lgb_result.sharpe_ratio:.2f}")

    return lgb_result


def compare_strategies(
    results_dir: str = "../results", charts_dir: str = "../results/charts"
):
    """
    기존 전략과 LightGBM 전략의 성능 비교

    Args:
        results_dir: 결과 저장 디렉토리
        charts_dir: 차트 저장 디렉토리
    """
    # 디렉토리 생성
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # 기존 백테스트 결과 로드
    try:
        original_results = pd.read_csv(
            os.path.join(results_dir, "individual_results.csv")
        )
        print("\n=== Loaded existing backtest results ===")
        print(
            original_results[
                [
                    "strategy_name",
                    "total_return_pct",
                    "sharpe_ratio",
                    "max_drawdown_pct",
                ]
            ].to_string(index=False)
        )
    except:
        print("\n=== Could not load existing results, generating new data ===")
        original_results = None

    # 샘플 데이터 생성 또는 로드
    try:
        data = pd.read_csv(
            os.path.join(results_dir, "sample_data.csv"), index_col=0, parse_dates=True
        )
        print(f"\nLoaded sample data with {len(data)} rows")
    except:
        data = generate_sample_data(start_date="2018-01-01", end_date="2022-12-31")
        data.to_csv(os.path.join(results_dir, "sample_data_new.csv"))
        print(f"\nGenerated new sample data with {len(data)} rows")

    # LightGBM 전략 실행
    lgb_result = run_lightgbm_strategy(data)

    # LightGBM 포트폴리오 가치 저장
    lgb_result.portfolio_value.to_csv(
        os.path.join(results_dir, "LightGBM_portfolio.csv")
    )

    # 기존 전략 결과가 없을 경우, 필요한 전략 실행
    if original_results is None:
        # 백테스터 초기화
        initial_capital = 10000.0
        commission = 0.001
        backtester = Backtester(data, initial_capital, commission)

        # 개별 전략 실행
        ma_params = {"ma_short": 20, "ma_long": 50}
        ma_strategy = MovingAverageCrossStrategy("MovingAverage", ma_params)
        ma_result = backtester.run(ma_strategy)

        rsi_params = {"rsi_period": 14, "oversold": 30, "overbought": 70}
        rsi_strategy = RSIStrategy("RSI", rsi_params)
        rsi_result = backtester.run(rsi_strategy)

        macd_params = {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9}
        macd_strategy = MACDStrategy("MACD", macd_params)
        macd_result = backtester.run(macd_strategy)

        # 복합 전략 실행
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

        # 결과 저장
        strategies_results = [ma_result, rsi_result, macd_result, composite_result]

        # 각 전략의 포트폴리오 가치 저장
        for result in strategies_results:
            result.portfolio_value.to_csv(
                os.path.join(results_dir, f"{result.strategy_name}_portfolio.csv")
            )
    else:
        # 기존 전략으로 다시 백테스트
        initial_capital = 10000.0
        commission = 0.001
        backtester = Backtester(data, initial_capital, commission)

        strategies_results = []

        # 원본 결과에서 전략 이름과 파라미터 가져오기
        for _, row in original_results.iterrows():
            strategy_name = row["strategy_name"]

            # 필터링된 파라미터만 가져오기 (param_ 접두어가 있는 컬럼)
            param_columns = [col for col in row.index if col.startswith("param_")]
            params = {col.replace("param_", ""): row[col] for col in param_columns}

            # 전략 객체 생성 및 백테스트
            if strategy_name == "MovingAverage":
                strategy = MovingAverageCrossStrategy(strategy_name, params)
            elif strategy_name == "RSI":
                strategy = RSIStrategy(strategy_name, params)
            elif strategy_name == "MACD":
                strategy = MACDStrategy(strategy_name, params)
            elif strategy_name == "Composite":
                strategy = create_composite_strategy(strategy_name, params)
            else:
                print(f"Unknown strategy: {strategy_name}, skipping...")
                continue

            result = backtester.run(strategy)
            strategies_results.append(result)

            # 포트폴리오 가치 저장
            result.portfolio_value.to_csv(
                os.path.join(results_dir, f"{strategy_name}_portfolio.csv")
            )

    # LightGBM 결과 추가
    all_results = strategies_results + [lgb_result]

    # 결과 CSV 저장
    comparison_df = save_results_to_csv(
        all_results, os.path.join(results_dir, "lightgbm_comparison.csv")
    )

    # 벤치마크 데이터 생성 (Buy & Hold)
    initial_capital = 10000.0
    benchmark = pd.Series(
        data["close"] / data["close"].iloc[0] * initial_capital, index=data.index
    )

    # 벤치마크 저장
    benchmark.to_csv(os.path.join(results_dir, "benchmark_portfolio.csv"))

    # LightGBM 전략 성과 시각화
    fig = lgb_result.plot_portfolio_performance(
        save_path=os.path.join(charts_dir, "LightGBM_performance.png")
    )
    plt.close(fig)

    fig = lgb_result.plot_cumulative_returns(
        benchmark=benchmark, save_path=os.path.join(charts_dir, "LightGBM_returns.png")
    )
    plt.close(fig)

    # 기존 전략과 LightGBM 전략 비교 차트
    plt.figure(figsize=(12, 8))

    # LightGBM 누적 수익률
    lgb_returns = (lgb_result.portfolio_value / initial_capital - 1) * 100
    plt.plot(
        lgb_returns.index,
        lgb_returns,
        "g-",
        linewidth=2,
        label=f"LightGBM ({lgb_returns.iloc[-1]:.2f}%)",
    )

    # 기존 전략 포트폴리오 데이터 로드 및 표시
    try:
        for result in strategies_results:
            portfolio_returns = (result.portfolio_value / initial_capital - 1) * 100
            plt.plot(
                portfolio_returns.index,
                portfolio_returns,
                label=f"{result.strategy_name} ({portfolio_returns.iloc[-1]:.2f}%)",
            )
    except Exception as e:
        print(f"Error plotting strategy results: {str(e)}")

    # 벤치마크 수익률
    benchmark_returns = (benchmark / initial_capital - 1) * 100
    plt.plot(
        benchmark_returns.index,
        benchmark_returns,
        "k--",
        label=f"Buy & Hold ({benchmark_returns.iloc[-1]:.2f}%)",
    )

    plt.title("Strategy Comparison: LightGBM vs Other Strategies")
    plt.ylabel("Cumulative Return (%)")
    plt.xlabel("Date")
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.2)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(
        os.path.join(charts_dir, "lightgbm_comparison.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    # 성과 메트릭스 요약 테이블
    comparison_metrics = comparison_df[
        [
            "strategy_name",
            "total_return_pct",
            "annual_return_pct",
            "sharpe_ratio",
            "max_drawdown_pct",
            "win_rate_pct",
            "num_trades",
        ]
    ]

    # 결과 출력
    print("\n=== Strategy Performance Comparison ===")
    print(comparison_metrics.to_string(index=False))

    # 결과 저장 (JSON)
    comparison_metrics.to_json(
        os.path.join(results_dir, "strategy_metrics.json"), orient="records"
    )

    print("\n=== Comparison completed successfully ===")
    print(f"Results saved to {results_dir}")
    print(f"Charts saved to {charts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare trading strategies performance"
    )
    parser.add_argument(
        "--results", type=str, default="../results", help="Results directory"
    )
    parser.add_argument(
        "--charts", type=str, default="../results/charts", help="Charts directory"
    )

    args = parser.parse_args()

    compare_strategies(results_dir=args.results, charts_dir=args.charts)
