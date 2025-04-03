"""
백테스팅 템플릿 파일

트레이딩 전략의 백테스팅을 위한 기본 템플릿입니다.
"""

import os
import sys
import argparse
import logging
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Any, Optional, Union

# 상위 디렉토리를 시스템 경로에 추가하여 src 모듈을 임포트할 수 있도록 합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils import setup_logger, ensure_directory, save_json, timestamp_string


class Strategy:
    """트레이딩 전략 추상 클래스"""

    def __init__(self, name: str, parameters: Dict[str, Union[int, float, str]]):
        """
        Args:
            name: 전략 이름
            parameters: 전략 파라미터
        """
        self.name = name
        self.parameters = parameters

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """데이터를 바탕으로 매수/매도 신호를 생성합니다.

        Args:
            data: OHLCV 데이터

        Returns:
            신호 시리즈 (1: 매수, -1: 매도, 0: 홀드)
        """
        raise NotImplementedError("하위 클래스에서 구현해야 합니다")


class MovingAverageCrossoverStrategy(Strategy):
    """이동평균선 교차 전략"""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """이동평균선 교차 기반 매매 신호 생성

        Args:
            data: OHLCV 데이터

        Returns:
            신호 시리즈 (1: 매수, -1: 매도, 0: 홀드)
        """
        # 파라미터 추출
        short_window = self.parameters.get("ma_short", 20)
        long_window = self.parameters.get("ma_long", 50)

        # 이동평균 계산
        df = data.copy()
        df["ma_short"] = df["close"].rolling(window=short_window).mean()
        df["ma_long"] = df["close"].rolling(window=long_window).mean()

        # 신호 초기화 (0: 홀드)
        signals = pd.Series(0, index=df.index)

        # 단기 이동평균이 장기 이동평균을 상향 돌파 (매수 신호: 1)
        signals[
            (df["ma_short"] > df["ma_long"])
            & (df["ma_short"].shift(1) <= df["ma_long"].shift(1))
        ] = 1

        # 단기 이동평균이 장기 이동평균을 하향 돌파 (매도 신호: -1)
        signals[
            (df["ma_short"] < df["ma_long"])
            & (df["ma_short"].shift(1) >= df["ma_long"].shift(1))
        ] = -1

        return signals


class BacktestResult:
    """백테스트 결과 클래스"""

    def __init__(
        self,
        strategy_name: str,
        signals: pd.Series,
        data: pd.DataFrame,
        initial_capital: float,
        positions: pd.Series,
        portfolio_value: pd.Series,
        benchmark: Optional[pd.Series] = None,
    ):
        """
        Args:
            strategy_name: 전략 이름
            signals: 생성된 매매 신호
            data: OHLCV 데이터
            initial_capital: 초기 자본금
            positions: 포지션 시리즈
            portfolio_value: 포트폴리오 가치 시리즈
            benchmark: 벤치마크 시리즈 (선택적)
        """
        self.strategy_name = strategy_name
        self.signals = signals
        self.data = data
        self.initial_capital = initial_capital
        self.positions = positions
        self.portfolio_value = portfolio_value
        self.benchmark = benchmark

        # 결과 지표 계산
        self.calculate_metrics()

    def calculate_metrics(self):
        """백테스트 결과 지표를 계산합니다."""
        # 최종 포트폴리오 가치
        self.final_value = self.portfolio_value.iloc[-1]

        # 총 수익률
        self.total_return = (self.final_value / self.initial_capital - 1) * 100

        # 연간 수익률
        days = (self.data.index[-1] - self.data.index[0]).days
        self.annual_return = ((1 + self.total_return / 100) ** (365 / days) - 1) * 100

        # 일일 수익률
        self.daily_returns = self.portfolio_value.pct_change().fillna(0)

        # 변동성
        self.volatility = self.daily_returns.std() * np.sqrt(252) * 100

        # 최대 손실 (Drawdown)
        rolling_max = self.portfolio_value.expanding().max()
        drawdown = (self.portfolio_value / rolling_max - 1) * 100
        self.max_drawdown = drawdown.min()

        # 샤프 비율
        risk_free_rate = 0.02  # 2% 기준 무위험 수익률
        daily_risk_free = ((1 + risk_free_rate) ** (1 / 252)) - 1
        self.sharpe_ratio = (
            (self.daily_returns.mean() - daily_risk_free)
            / self.daily_returns.std()
            * np.sqrt(252)
        )

        # 거래 수 및 승률 계산
        trades = self.signals[self.signals != 0]
        self.num_trades = len(trades)

        # 수익 거래 계산 (신호 발생 시점부터 다음 신호까지의 수익)
        if self.num_trades > 0:
            returns = []
            entry_price = None
            position = 0

            for i, (date, signal) in enumerate(trades.items()):
                price = self.data.loc[date, "close"]

                if signal == 1:  # 매수 신호
                    entry_price = price
                    position = 1
                elif signal == -1 and position == 1:  # 매도 신호 (보유 중인 경우)
                    if entry_price is not None:
                        returns.append((price / entry_price - 1) * 100)
                    entry_price = None
                    position = 0

            # 마지막 포지션이 청산되지 않은 경우
            if position == 1 and entry_price is not None:
                last_price = self.data["close"].iloc[-1]
                returns.append((last_price / entry_price - 1) * 100)

            # 승률 계산
            if returns:
                self.win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
            else:
                self.win_rate = 0
        else:
            self.win_rate = 0

        # 수익/위험 비율
        if self.max_drawdown != 0:
            self.return_risk_ratio = abs(self.total_return / self.max_drawdown)
        else:
            self.return_risk_ratio = float("inf")

    def plot_portfolio_performance(self, save_path: Optional[str] = None) -> plt.Figure:
        """포트폴리오 성과를 시각화합니다.

        Args:
            save_path: 차트 저장 경로 (선택적)

        Returns:
            생성된 Figure 객체
        """
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 16), gridspec_kw={"height_ratios": [3, 1, 1]}
        )

        # 가격 및 포트폴리오 가치 플롯
        ax1.plot(self.data.index, self.data["close"], "k-", label="Price")
        ax1_twin = ax1.twinx()
        ax1_twin.plot(
            self.portfolio_value.index,
            self.portfolio_value,
            "g-",
            label="Portfolio Value",
        )

        # 매수/매도 신호 표시
        buy_signals = self.signals[self.signals == 1]
        sell_signals = self.signals[self.signals == -1]

        for date in buy_signals.index:
            ax1.axvline(x=date, color="g", linestyle="--", alpha=0.3)

        for date in sell_signals.index:
            ax1.axvline(x=date, color="r", linestyle="--", alpha=0.3)

        # 포지션 플롯
        ax2.plot(self.positions.index, self.positions, "b-", label="Position")
        ax2.set_ylabel("Position")
        ax2.set_ylim(-0.1, 1.1)

        # Drawdown 플롯
        rolling_max = self.portfolio_value.expanding().max()
        drawdown = (self.portfolio_value / rolling_max - 1) * 100
        ax3.fill_between(drawdown.index, drawdown, 0, color="r", alpha=0.3)
        ax3.set_ylabel("Drawdown (%)")

        # 레이블 및 범례 설정
        ax1.set_title(
            f"{self.strategy_name} - Portfolio Performance\nReturn: {self.total_return:.2f}%, Sharpe: {self.sharpe_ratio:.2f}"
        )
        ax1.set_ylabel("Price")
        ax1_twin.set_ylabel("Portfolio Value")
        ax1.legend(loc="upper left")
        ax1_twin.legend(loc="upper right")

        plt.tight_layout()

        if save_path:
            ensure_directory(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def to_dict(self) -> Dict:
        """백테스트 결과를 딕셔너리로 변환합니다."""
        return {
            "strategy_name": self.strategy_name,
            "initial_capital": self.initial_capital,
            "final_value": self.final_value,
            "total_return_pct": self.total_return,
            "annual_return_pct": self.annual_return,
            "volatility": self.volatility,
            "max_drawdown_pct": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "num_trades": self.num_trades,
            "win_rate_pct": self.win_rate,
            "return_risk_ratio": self.return_risk_ratio,
        }


class Backtester:
    """백테스팅 엔진 클래스"""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
        benchmark_fn: Optional[Callable] = None,
    ):
        """
        Args:
            data: OHLCV 데이터프레임 (컬럼: open, high, low, close, volume)
            initial_capital: 초기 자본금
            commission: 거래 수수료 비율 (0.001 = 0.1%)
            benchmark_fn: 벤치마크 생성 함수 (선택적)
        """
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.benchmark_fn = benchmark_fn

        # 벤치마크 생성
        if self.benchmark_fn:
            self.benchmark = self.benchmark_fn(data, initial_capital)
        else:
            # 기본 벤치마크: Buy & Hold
            self.benchmark = pd.Series(
                data["close"] / data["close"].iloc[0] * initial_capital,
                index=data.index,
            )

    def run(self, strategy: Strategy) -> BacktestResult:
        """전략에 대해 백테스트를 실행합니다.

        Args:
            strategy: 백테스트할 전략 객체

        Returns:
            백테스트 결과 객체
        """
        # 매매 신호 생성
        signals = strategy.generate_signals(self.data)

        # 포지션 및 포트폴리오 가치 초기화
        positions = pd.Series(0, index=self.data.index)
        portfolio_value = pd.Series(self.initial_capital, index=self.data.index)

        # 첫 번째 날은 신호에 따라 매수/매도 불가능
        cash = self.initial_capital
        shares = 0

        # 각 일자에 대해 포트폴리오 가치 계산
        for i, date in enumerate(self.data.index[1:], 1):
            current_price = self.data.loc[date, "close"]
            prev_date = self.data.index[i - 1]

            # 매매 신호에 따라 포지션 변경
            if signals.loc[prev_date] == 1 and shares == 0:  # 매수 신호
                shares = cash * (1 - self.commission) / current_price
                cash = 0
                positions.loc[date:] = 1
            elif signals.loc[prev_date] == -1 and shares > 0:  # 매도 신호
                cash = shares * current_price * (1 - self.commission)
                shares = 0
                positions.loc[date:] = 0

            # 포트폴리오 가치 계산
            portfolio_value.loc[date] = cash + shares * current_price

        # 백테스트 결과 반환
        return BacktestResult(
            strategy.name,
            signals,
            self.data,
            self.initial_capital,
            positions,
            portfolio_value,
            self.benchmark,
        )


def parse_args():
    """명령줄 인자 파싱 함수"""
    parser = argparse.ArgumentParser(description="백테스트 실행 스크립트")
    parser.add_argument("--data", type=str, required=True, help="데이터 파일 경로")
    parser.add_argument(
        "--results", type=str, default="../results", help="결과 저장 디렉토리"
    )
    parser.add_argument("--capital", type=float, default=10000.0, help="초기 자본금")
    parser.add_argument(
        "--commission", type=float, default=0.001, help="거래 수수료 (0.001 = 0.1%)"
    )
    parser.add_argument("--ma_short", type=int, default=20, help="단기 이동평균 기간")
    parser.add_argument("--ma_long", type=int, default=50, help="장기 이동평균 기간")
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()

    # 결과 디렉토리 생성
    results_dir = args.results
    timestamp = timestamp_string()
    experiment_dir = os.path.join(results_dir, f"backtest_{timestamp}")
    ensure_directory(experiment_dir)

    # 로거 설정
    log_file = os.path.join(experiment_dir, "backtest.log")
    logger = setup_logger("backtest", log_file)

    logger.info("백테스트 시작")
    logger.info(f"데이터 파일: {args.data}")
    logger.info(f"초기 자본금: {args.capital}")
    logger.info(f"거래 수수료: {args.commission * 100}%")

    try:
        # 데이터 로드
        data = pd.read_csv(args.data, index_col=0, parse_dates=True)
        logger.info(f"데이터 로드 완료: {len(data)} 행")

        # 백테스터 초기화
        backtester = Backtester(data, args.capital, args.commission)

        # 이동평균 교차 전략 실행
        ma_params = {"ma_short": args.ma_short, "ma_long": args.ma_long}
        ma_strategy = MovingAverageCrossoverStrategy("MA_Crossover", ma_params)

        logger.info(f"전략 실행: {ma_strategy.name}")
        logger.info(f"파라미터: {ma_params}")

        # 백테스트 실행
        result = backtester.run(ma_strategy)

        # 결과 저장
        result_dict = result.to_dict()
        result_file = os.path.join(experiment_dir, "results.json")
        save_json(result_dict, result_file)

        # 포트폴리오 가치 저장
        portfolio_file = os.path.join(experiment_dir, "portfolio.csv")
        result.portfolio_value.to_csv(portfolio_file)

        # 성과 시각화
        chart_file = os.path.join(experiment_dir, "performance.png")
        result.plot_portfolio_performance(chart_file)

        # 결과 출력
        logger.info("백테스트 완료")
        logger.info(f"총 수익률: {result.total_return:.2f}%")
        logger.info(f"연간 수익률: {result.annual_return:.2f}%")
        logger.info(f"샤프 비율: {result.sharpe_ratio:.2f}")
        logger.info(f"최대 손실: {result.max_drawdown:.2f}%")
        logger.info(f"거래 횟수: {result.num_trades}")
        logger.info(f"승률: {result.win_rate:.2f}%")
        logger.info(f"결과 저장 경로: {experiment_dir}")

        return 0
    except Exception as e:
        logger.error(f"백테스트 중 오류 발생: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
