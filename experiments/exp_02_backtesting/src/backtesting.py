import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from typing import Dict, List, Tuple, Callable, Optional, Union
import time
import json


class Strategy:
    """트레이딩 전략 클래스"""

    def __init__(self, name: str, parameters: Dict[str, Union[int, float, str]]):
        """
        Args:
            name: 전략 이름
            parameters: 전략 파라미터 (예: {'ma_short': 10, 'ma_long': 30, 'rsi_period': 14})
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
        raise NotImplementedError("자식 클래스에서 구현해야 합니다")

    def __str__(self) -> str:
        return f"Strategy(name={self.name}, params={self.parameters})"


class MovingAverageCrossStrategy(Strategy):
    """이동평균선 교차 전략"""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """이동평균선 교차 기반 매매 신호 생성

        Args:
            data: OHLCV 데이터

        Returns:
            신호 시리즈 (1: 매수, -1: 매도, 0: 홀드)
        """
        # 파라미터 추출
        short_window = self.parameters.get("ma_short", 10)
        long_window = self.parameters.get("ma_long", 30)

        # 단기 및 장기 이동평균 계산
        data["ma_short"] = data["close"].rolling(window=short_window).mean()
        data["ma_long"] = data["close"].rolling(window=long_window).mean()

        # 신호 초기화
        signals = pd.Series(0, index=data.index)

        # 매수 신호: 단기 이동평균이 장기 이동평균을 상향 돌파
        signals[data["ma_short"] > data["ma_long"]] = 1

        # 매도 신호: 단기 이동평균이 장기 이동평균을 하향 돌파
        signals[data["ma_short"] < data["ma_long"]] = -1

        return signals


class RSIStrategy(Strategy):
    """RSI(Relative Strength Index) 기반 전략"""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """RSI 기반 매매 신호 생성

        Args:
            data: OHLCV 데이터

        Returns:
            신호 시리즈 (1: 매수, -1: 매도, 0: 홀드)
        """
        # 파라미터 추출
        period = self.parameters.get("rsi_period", 14)
        oversold = self.parameters.get("oversold", 30)
        overbought = self.parameters.get("overbought", 70)

        # RSI 계산
        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        data["rsi"] = rsi

        # 신호 초기화
        signals = pd.Series(0, index=data.index)

        # 매수 신호: RSI가 과매도 구간에서 상승
        signals[(data["rsi"] < oversold) & (data["rsi"].shift(1) >= oversold)] = 1

        # 매도 신호: RSI가 과매수 구간에서 하락
        signals[(data["rsi"] > overbought) & (data["rsi"].shift(1) <= overbought)] = -1

        return signals


class MACDStrategy(Strategy):
    """MACD(Moving Average Convergence Divergence) 기반 전략"""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """MACD 기반 매매 신호 생성

        Args:
            data: OHLCV 데이터

        Returns:
            신호 시리즈 (1: 매수, -1: 매도, 0: 홀드)
        """
        # 파라미터 추출
        fast_period = self.parameters.get("macd_fast", 12)
        slow_period = self.parameters.get("macd_slow", 26)
        signal_period = self.parameters.get("macd_signal", 9)

        # MACD 계산
        ema_fast = data["close"].ewm(span=fast_period).mean()
        ema_slow = data["close"].ewm(span=slow_period).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()

        data["macd"] = macd_line
        data["macd_signal"] = signal_line
        data["macd_histogram"] = macd_line - signal_line

        # 신호 초기화
        signals = pd.Series(0, index=data.index)

        # 매수 신호: MACD 라인이 시그널 라인을 상향 돌파
        signals[
            (data["macd"] > data["macd_signal"])
            & (data["macd"].shift(1) <= data["macd_signal"].shift(1))
        ] = 1

        # 매도 신호: MACD 라인이 시그널 라인을 하향 돌파
        signals[
            (data["macd"] < data["macd_signal"])
            & (data["macd"].shift(1) >= data["macd_signal"].shift(1))
        ] = -1

        return signals


class CompositeStrategy(Strategy):
    """여러 전략을 결합한 복합 전략"""

    def __init__(
        self,
        name: str,
        parameters: Dict[str, Union[int, float, str]],
        strategies: List[Strategy],
        weights: Optional[List[float]] = None,
    ):
        """
        Args:
            name: 전략 이름
            parameters: 전략 파라미터
            strategies: 결합할 전략 리스트
            weights: 각 전략의 가중치 (기본값: 동일 가중치)
        """
        super().__init__(name, parameters)
        self.strategies = strategies

        if weights is None:
            # 동일 가중치 할당
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            assert len(weights) == len(
                strategies
            ), "전략과 가중치 개수가 일치해야 합니다"
            # 가중치 정규화
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """여러 전략의 신호를 결합하여 최종 신호 생성

        Args:
            data: OHLCV 데이터

        Returns:
            신호 시리즈 (1: 매수, -1: 매도, 0: 홀드)
        """
        # 각 전략의 신호 계산
        all_signals = []
        for strategy, weight in zip(self.strategies, self.weights):
            signals = strategy.generate_signals(data)
            all_signals.append(signals * weight)

        # 신호 결합
        combined_signals = pd.Series(0, index=data.index)
        for signal in all_signals:
            combined_signals += signal

        # 신호 이산화 (threshold 기반)
        threshold = self.parameters.get("threshold", 0.5)

        result = pd.Series(0, index=data.index)
        result[combined_signals > threshold] = 1
        result[combined_signals < -threshold] = -1

        return result


class BacktestResult:
    """백테스트 결과를 저장하고 분석하는 클래스"""

    def __init__(
        self,
        strategy_name: str,
        parameters: Dict[str, Union[int, float, str]],
        initial_capital: float,
        positions: pd.Series,
        portfolio_value: pd.Series,
        signals: pd.Series,
        data: pd.DataFrame,
        trades: List[Dict],
    ):
        """
        Args:
            strategy_name: 전략 이름
            parameters: 사용된 파라미터
            initial_capital: 초기 자본
            positions: 포지션 시리즈
            portfolio_value: 포트폴리오 가치 시리즈
            signals: 신호 시리즈
            data: 원본 가격 데이터
            trades: 매매 이력 리스트
        """
        self.strategy_name = strategy_name
        self.parameters = parameters
        self.initial_capital = initial_capital
        self.positions = positions
        self.portfolio_value = portfolio_value
        self.signals = signals
        self.data = data
        self.trades = trades

        # 성과 지표 계산
        self._calculate_metrics()

    def _calculate_metrics(self):
        """다양한 성과 지표를 계산합니다."""
        # 수익률 계산
        self.total_return = (
            self.portfolio_value.iloc[-1] / self.initial_capital - 1
        ) * 100

        # 연간 수익률 (CAGR)
        days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
        self.annual_return = (
            ((1 + self.total_return / 100) ** (365 / days) - 1) * 100 if days > 0 else 0
        )

        # 거래 횟수
        self.num_trades = len(self.trades)

        # 승률 계산
        if self.num_trades > 0:
            winning_trades = [t for t in self.trades if t["profit_pct"] > 0]
            self.win_rate = len(winning_trades) / self.num_trades * 100
        else:
            self.win_rate = 0

        # 일일 수익률
        self.daily_returns = self.portfolio_value.pct_change().fillna(0) * 100

        # 변동성 (연간화된 표준편차)
        self.volatility = self.daily_returns.std() * (252**0.5)

        # 최대 낙폭 (MDD)
        rolling_max = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value / rolling_max - 1) * 100
        self.max_drawdown = drawdown.min()

        # 샤프 비율
        risk_free_rate = 0.02  # 2% 연간 무위험 수익률 가정
        daily_risk_free = risk_free_rate / 252
        self.sharpe_ratio = (
            (self.daily_returns.mean() - daily_risk_free)
            / self.daily_returns.std()
            * (252**0.5)
            if self.daily_returns.std() != 0
            else 0
        )

        # 수익 대비 위험 (Return/Risk Ratio)
        self.return_risk_ratio = (
            self.annual_return / self.volatility if self.volatility != 0 else 0
        )

    def to_dict(self) -> Dict:
        """결과를 딕셔너리로 반환합니다."""
        return {
            "strategy_name": self.strategy_name,
            "parameters": self.parameters,
            "initial_capital": self.initial_capital,
            "final_value": self.portfolio_value.iloc[-1],
            "total_return_pct": self.total_return,
            "annual_return_pct": self.annual_return,
            "num_trades": self.num_trades,
            "win_rate_pct": self.win_rate,
            "volatility": self.volatility,
            "max_drawdown_pct": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "return_risk_ratio": self.return_risk_ratio,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """결과를 데이터프레임으로 변환합니다."""
        result_dict = self.to_dict()
        # 파라미터 항목들을 별도 컬럼으로 추가
        for key, value in self.parameters.items():
            result_dict[f"param_{key}"] = value

        return pd.DataFrame([result_dict])

    def plot_portfolio_performance(self, save_path: Optional[str] = None) -> plt.Figure:
        """포트폴리오 성과를 시각화합니다.

        Args:
            save_path: 이미지 저장 경로 (None인 경우 저장하지 않음)

        Returns:
            생성된 matplotlib Figure 객체
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

        # 가격 및 포트폴리오 가치 그래프
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data["close"], "b-", label="Price")
        ax1_right = ax1.twinx()
        ax1_right.plot(
            self.portfolio_value.index,
            self.portfolio_value,
            "g-",
            label="Portfolio Value",
        )

        # 매수/매도 지점 표시
        buy_signals = self.signals[self.signals == 1].index
        sell_signals = self.signals[self.signals == -1].index

        ax1.plot(
            buy_signals,
            self.data.loc[buy_signals, "close"],
            "^",
            markersize=10,
            color="g",
            label="Buy",
        )
        ax1.plot(
            sell_signals,
            self.data.loc[sell_signals, "close"],
            "v",
            markersize=10,
            color="r",
            label="Sell",
        )

        ax1.set_title(f"Portfolio Performance - {self.strategy_name}")
        ax1.set_ylabel("Price")
        ax1_right.set_ylabel("Portfolio Value")
        ax1.legend(loc="upper left")
        ax1_right.legend(loc="upper right")

        # 포지션 그래프
        ax2 = axes[1]
        ax2.plot(self.positions.index, self.positions, "k-", label="Position")
        ax2.set_ylabel("Position Size")
        ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax2.legend()

        # 낙폭 그래프
        ax3 = axes[2]
        rolling_max = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value / rolling_max - 1) * 100
        ax3.fill_between(drawdown.index, drawdown, 0, color="r", alpha=0.3)
        ax3.set_ylabel("Drawdown (%)")
        ax3.set_xlabel("Date")
        ax3.axhline(
            y=self.max_drawdown,
            color="r",
            linestyle="--",
            label=f"Max Drawdown: {self.max_drawdown:.2f}%",
        )
        ax3.legend()

        # 정보 테이블 추가
        info_text = (
            f"Total Return: {self.total_return:.2f}%\n"
            f"Annual Return: {self.annual_return:.2f}%\n"
            f"Volatility: {self.volatility:.2f}%\n"
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            f"Max Drawdown: {self.max_drawdown:.2f}%\n"
            f"Win Rate: {self.win_rate:.2f}%\n"
            f"Trades: {self.num_trades}"
        )

        fig.text(
            0.02,
            0.02,
            info_text,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        fig.tight_layout()

        # 결과 저장
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        return fig

    def plot_cumulative_returns(
        self, benchmark: Optional[pd.Series] = None, save_path: Optional[str] = None
    ) -> plt.Figure:
        """누적 수익률을 시각화합니다.

        Args:
            benchmark: 벤치마크 수익률 시리즈 (None인 경우 표시하지 않음)
            save_path: 이미지 저장 경로 (None인 경우 저장하지 않음)

        Returns:
            생성된 matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # 전략 누적 수익률
        portfolio_cum_returns = (self.portfolio_value / self.initial_capital - 1) * 100
        ax.plot(
            portfolio_cum_returns.index,
            portfolio_cum_returns,
            "b-",
            label=f"{self.strategy_name} ({self.total_return:.2f}%)",
        )

        # 벤치마크가 제공된 경우
        if benchmark is not None:
            benchmark_cum_returns = (benchmark / benchmark.iloc[0] - 1) * 100
            ax.plot(
                benchmark_cum_returns.index,
                benchmark_cum_returns,
                "r--",
                label=f"Benchmark ({benchmark_cum_returns.iloc[-1]:.2f}%)",
            )

        ax.set_title("Cumulative Returns")
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_xlabel("Date")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 파라미터 정보 추가
        param_text = "Parameters:\n" + "\n".join(
            [f"{key}: {value}" for key, value in self.parameters.items()]
        )
        fig.text(
            0.02,
            0.02,
            param_text,
            ha="left",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        fig.tight_layout()

        # 결과 저장
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        return fig


class Backtester:
    """백테스팅 시뮬레이터 클래스"""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        commission: float = 0.001,
    ):
        """
        Args:
            data: OHLCV 데이터 (datetime 인덱스, 'open', 'high', 'low', 'close', 'volume' 컬럼 필요)
            initial_capital: 초기 자본금
            commission: 거래 수수료 (거래 가격 대비 비율)
        """
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission

    def run(self, strategy: Strategy) -> BacktestResult:
        """전략을 실행하여 백테스트 수행

        Args:
            strategy: 백테스트할 전략 객체

        Returns:
            백테스트 결과 객체
        """
        # 신호 생성
        signals = strategy.generate_signals(self.data)

        # 포트폴리오 초기화
        portfolio = pd.Series(index=self.data.index)
        portfolio.iloc[0] = self.initial_capital

        # 포지션 초기화 (보유 수량)
        position = pd.Series(0, index=self.data.index)

        # 거래 이력
        trades = []
        entry_price = 0
        entry_date = None

        # 백테스트 실행
        for i in range(1, len(self.data)):
            current_date = self.data.index[i]
            prev_date = self.data.index[i - 1]

            # 전날 포트폴리오 가치 가져오기
            portfolio[current_date] = portfolio[prev_date]

            # 전날 포지션 유지
            position[current_date] = position[prev_date]

            # 신호에 따른 거래 실행
            if (
                signals[prev_date] == 1 and position[current_date] == 0
            ):  # 매수 신호 & 현재 포지션 없음
                # 매수 가격 (당일 시가)
                price = self.data.loc[current_date, "open"]

                # 수수료를 고려한 실제 매수 가능 수량
                available_capital = portfolio[current_date]
                position_size = available_capital / price * (1 - self.commission)

                # 포지션 업데이트
                position[current_date] = position_size

                # 포트폴리오 가치 업데이트 (수수료 차감)
                commission_cost = position_size * price * self.commission
                portfolio[current_date] -= commission_cost

                # 거래 기록
                entry_price = price
                entry_date = current_date

            elif (
                signals[prev_date] == -1 and position[current_date] > 0
            ):  # 매도 신호 & 현재 포지션 있음
                # 매도 가격 (당일 시가)
                price = self.data.loc[current_date, "open"]

                # 현재 포지션 사이즈
                position_size = position[current_date]

                # 매도 금액 (수수료 차감)
                sell_value = position_size * price
                commission_cost = sell_value * self.commission
                net_sell_value = sell_value - commission_cost

                # 포트폴리오 가치 업데이트
                portfolio[current_date] = net_sell_value

                # 포지션 정리
                position[current_date] = 0

                # 거래 기록
                profit = (price - entry_price) * position_size - commission_cost
                profit_pct = (price / entry_price - 1) * 100

                trades.append(
                    {
                        "entry_date": entry_date,
                        "entry_price": entry_price,
                        "exit_date": current_date,
                        "exit_price": price,
                        "position_size": position_size,
                        "profit": profit,
                        "profit_pct": profit_pct,
                    }
                )

            # 매일 종가 기준 포트폴리오 가치 업데이트
            if position[current_date] > 0:
                close_price = self.data.loc[current_date, "close"]
                portfolio[current_date] = position[current_date] * close_price

        # 백테스트 결과 생성
        result = BacktestResult(
            strategy_name=strategy.name,
            parameters=strategy.parameters,
            initial_capital=self.initial_capital,
            positions=position,
            portfolio_value=portfolio,
            signals=signals,
            data=self.data,
            trades=trades,
        )

        return result


def parameter_grid_search(
    backtester: Backtester,
    strategy_class,
    param_grid: Dict[str, List],
    strategy_name: str = "GridSearch",
) -> Tuple[List[BacktestResult], BacktestResult]:
    """파라미터 그리드 탐색을 수행합니다

    Args:
        backtester: 백테스터 객체
        strategy_class: 전략 클래스
        param_grid: 파라미터 그리드 (예: {'ma_short': [5, 10, 15], 'ma_long': [20, 30]})
        strategy_name: 기본 전략 이름

    Returns:
        (모든 결과 리스트, 최적 결과)
    """
    print(f"Grid Search: {strategy_name}")
    print(f"Parameter grid: {param_grid}")

    # 파라미터 조합 생성
    import itertools

    param_keys = param_grid.keys()
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    total_combinations = len(param_combinations)
    print(f"Total combinations: {total_combinations}")

    # 모든 조합에 대해 백테스트 실행
    start_time = time.time()
    results = []

    for i, combo in enumerate(param_combinations):
        # 파라미터 딕셔너리 생성
        params = dict(zip(param_keys, combo))

        # 전략 인스턴스 생성
        strategy = strategy_class(f"{strategy_name}_{i+1}", params)

        # 백테스트 실행
        result = backtester.run(strategy)
        results.append(result)

        # 진행 상황 출력
        if (i + 1) % 10 == 0 or i + 1 == total_combinations:
            elapsed = time.time() - start_time
            print(
                f"Processed {i+1}/{total_combinations} combinations. Elapsed: {elapsed:.2f}s"
            )

    # 최적 결과 찾기 (Sharpe 비율 기준)
    best_result = max(results, key=lambda x: x.sharpe_ratio)

    print(f"\nBest result:")
    print(f"Parameters: {best_result.parameters}")
    print(f"Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
    print(f"Total Return: {best_result.total_return:.2f}%")

    return results, best_result


def save_results_to_csv(results: List[BacktestResult], filepath: str) -> pd.DataFrame:
    """백테스트 결과를 CSV 파일로 저장합니다

    Args:
        results: 백테스트 결과 리스트
        filepath: 저장할 파일 경로

    Returns:
        결과 데이터프레임
    """
    # 결과를 데이터프레임으로 변환
    result_dfs = [result.to_dataframe() for result in results]
    combined_df = pd.concat(result_dfs, ignore_index=True)

    # CSV 파일로 저장
    combined_df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")

    return combined_df
