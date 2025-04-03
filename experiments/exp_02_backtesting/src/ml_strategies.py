import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Union, Optional, Tuple
from backtesting import Strategy


class LightGBMStrategy(Strategy):
    """LightGBM 머신러닝 기반 트레이딩 전략"""

    def __init__(self, name: str, parameters: Dict[str, Union[int, float, str]]):
        """
        Args:
            name: 전략 이름
            parameters: 전략 파라미터
                - prediction_horizon: 예측 시점 (몇 일 후 수익률 예측, 기본값: 5)
                - train_interval: 모델 재학습 주기 (일, 기본값: 60)
                - features_lookback: 특성 계산을 위한 룩백 기간 (기본값: 20)
                - lgb_params: LightGBM 하이퍼파라미터 (None일 경우 기본값 사용)
                - threshold_buy: 매수 신호 임계값 (기본값: 0.1)
                - threshold_sell: 매도 신호 임계값 (기본값: -0.1)
        """
        super().__init__(name, parameters)
        self.model = None
        self.last_train_date = None
        self.train_features = None
        self.train_targets = None

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        가격 데이터로부터 특성(features) 생성

        Args:
            data: OHLCV 데이터

        Returns:
            특성 데이터프레임
        """
        df = data.copy()
        lookback = self.parameters.get("features_lookback", 20)

        # 기본 가격 관련 특성
        df["returns_1d"] = df["close"].pct_change(1)
        df["returns_5d"] = df["close"].pct_change(5)

        # 이동평균 관련 특성
        for window in [5, 10, 20, 50]:
            df[f"ma_{window}"] = df["close"].rolling(window=window).mean()
            # 현재 가격과 이동평균의 차이 (%)
            df[f"close_ma_{window}_diff"] = (df["close"] / df[f"ma_{window}"] - 1) * 100

        # 볼린저 밴드 관련 특성
        for window in [20]:
            mid = df["close"].rolling(window=window).mean()
            std = df["close"].rolling(window=window).std()
            df[f"bb_upper_{window}"] = mid + 2 * std
            df[f"bb_lower_{window}"] = mid - 2 * std
            # 밴드 내 위치 (0~1)
            df[f"bb_position_{window}"] = (df["close"] - df[f"bb_lower_{window}"]) / (
                df[f"bb_upper_{window}"] - df[f"bb_lower_{window}"]
            )

        # 거래량 관련 특성
        df["volume_change"] = df["volume"].pct_change(1)
        df["volume_ma_5"] = df["volume"].rolling(window=5).mean()
        df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma_20"]

        # RSI 계산
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # RSI 14
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # 가격 범위 관련 특성
        df["daily_range"] = (
            (df["high"] - df["low"]) / df["close"] * 100
        )  # 일일 변동폭 (%)
        df["daily_range_ma_5"] = df["daily_range"].rolling(window=5).mean()

        # 추세 강도 지표 (ADX 간소화 버전)
        df["tr"] = np.maximum(
            np.maximum(df["high"] - df["low"], abs(df["high"] - df["close"].shift(1))),
            abs(df["low"] - df["close"].shift(1)),
        )
        df["atr_14"] = df["tr"].rolling(window=14).mean()

        # 모멘텀 지표
        for window in [5, 10, 20]:
            df[f"momentum_{window}"] = df["close"] / df["close"].shift(window) - 1

        # NaN 값 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)

        # 특성 선택 (모든 계산된 특성 사용)
        features = df.columns.difference(["open", "high", "low", "close", "volume"])

        return df[features]

    def _create_target(self, data: pd.DataFrame) -> pd.Series:
        """
        학습 타겟 생성 (미래 가격 변화)

        Args:
            data: OHLCV 데이터

        Returns:
            타겟 시리즈 (미래 수익률)
        """
        horizon = self.parameters.get("prediction_horizon", 5)
        future_returns = data["close"].pct_change(horizon).shift(-horizon)
        return future_returns

    def _train_model(self, features: pd.DataFrame, targets: pd.Series) -> lgb.Booster:
        """
        LightGBM 모델 학습

        Args:
            features: 특성 데이터프레임
            targets: 타겟 시리즈

        Returns:
            학습된 LightGBM 모델
        """
        # 기본 LightGBM 파라미터
        default_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        # 사용자 지정 파라미터가 있을 경우 업데이트
        lgb_params = self.parameters.get("lgb_params", {})
        if lgb_params:
            default_params.update(lgb_params)

        # NaN 값 처리
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        targets = targets.fillna(0)

        # 학습 데이터셋 생성
        train_data = lgb.Dataset(features, label=targets)

        # 모델 학습
        num_rounds = self.parameters.get("num_boost_round", 100)
        model = lgb.train(default_params, train_data, num_boost_round=num_rounds)

        return model

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        LightGBM 모델을 사용하여 매매 신호 생성

        Args:
            data: OHLCV 데이터

        Returns:
            신호 시리즈 (1: 매수, -1: 매도, 0: 홀드)
        """
        # 신호 초기화
        signals = pd.Series(0, index=data.index)

        # 특성 생성
        features = self._create_features(data)

        # 타겟 생성 (학습용)
        targets = self._create_target(data)

        # 재학습 주기 설정
        train_interval = self.parameters.get("train_interval", 60)

        # 학습에 필요한 충분한 데이터가 있는지 확인
        min_required_data = max(
            50,
            self.parameters.get("features_lookback", 20)
            + self.parameters.get("prediction_horizon", 5),
        )

        # 학습에 필요한 데이터보다 적으면 신호 생성 건너뜀
        if len(data) < min_required_data:
            return signals

        # 각 날짜에 대해 처리
        for i, current_date in enumerate(data.index):
            # 학습 데이터 충분한지 확인
            if i < min_required_data:
                continue

            # 최초 모델 학습 또는 재학습 주기 도달 시 모델 재학습
            if (
                self.model is None
                or self.last_train_date is None
                or (current_date - self.last_train_date).days >= train_interval
            ):
                # 현재 시점까지의 데이터로 모델 학습
                train_features = features.iloc[:i]
                train_targets = targets.iloc[:i]

                # NaN 값 제거
                valid_indices = ~train_targets.isna()

                if (
                    valid_indices.sum() > min_required_data
                ):  # 충분한 유효 데이터가 있는지 확인
                    self.model = self._train_model(
                        train_features[valid_indices], train_targets[valid_indices]
                    )
                    self.last_train_date = current_date
                    self.train_features = train_features
                    self.train_targets = train_targets

            # 모델이 학습되었다면 예측 수행
            if self.model is not None:
                # 현재 시점의 특성으로 예측
                current_features = features.iloc[i : i + 1]

                # 예측 수행
                prediction = self.model.predict(current_features)[0]

                # 임계값 설정
                threshold_buy = self.parameters.get(
                    "threshold_buy", 0.1
                )  # 1% 상승 예상 시 매수
                threshold_sell = self.parameters.get(
                    "threshold_sell", -0.1
                )  # 1% 하락 예상 시 매도

                # 신호 생성
                if prediction > threshold_buy:
                    signals.iloc[i] = 1  # 매수 신호
                elif prediction < threshold_sell:
                    signals.iloc[i] = -1  # 매도 신호

        return signals
