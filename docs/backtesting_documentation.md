# 트레이딩 전략 백테스팅 시뮬레이터 문서

**작성일**: 2023-04-03
**관련 실험**: exp_02_backtesting

## 1. 개요

이 문서는 트레이딩 전략의 성능을 평가하기 위한 백테스팅 시뮬레이터의 설계와 구현에 대해 설명합니다. 이 시뮬레이터는 다양한 기술적 지표 기반 트레이딩 전략을 테스트하고 최적의 파라미터를 찾기 위한 도구로 사용할 수 있습니다.

백테스팅이란 과거 데이터를 사용하여 투자 전략의 유효성을 검증하는 과정으로, 이를 통해 전략의 수익성, 위험성, 안정성 등을 미리 평가할 수 있습니다.

## 2. 시스템 구성

### 2.1 주요 컴포넌트

백테스팅 시스템은 다음과 같은 주요 컴포넌트로 구성됩니다:

1. **Strategy 클래스**: 트레이딩 전략을 정의하는 추상 클래스
2. **Backtester 클래스**: 전략을 과거 데이터에 적용하여 백테스트를 수행하는 클래스
3. **BacktestResult 클래스**: 백테스트 결과를 저장하고 분석하는 클래스
4. **파라미터 최적화 도구**: 최적의 전략 파라미터를 탐색하는 유틸리티

### 2.2 파일 구조

```
experiments/exp_02_backtesting/
├── src/
│   ├── backtesting.py   # 핵심 백테스팅 엔진
│   └── run_backtest.py  # 백테스트 실행 스크립트
├── data/
│   └── (백테스트에 사용되는 데이터)
├── results/
│   ├── sample_data.csv  # 샘플 데이터
│   ├── individual_results.csv # 개별 전략 결과
│   ├── grid_search_results.csv # 그리드 탐색 결과
│   └── charts/         # 시각화 이미지 저장 폴더
└── report.md           # 실험 보고서
```

## 3. 핵심 기능 설명

### 3.1 트레이딩 전략 (Strategy)

#### 기본 전략 인터페이스
```python
class Strategy:
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters
    
    def generate_signals(self, data):
        # 상속 클래스에서 구현
        raise NotImplementedError()
```

#### 구현된 전략

1. **이동평균선 교차 전략 (MovingAverageCrossStrategy)**
   - 두 개의 이동평균선 (단기 및 장기)의 교차를 기반으로 매매 신호 생성
   - 파라미터: `ma_short` (단기 이동평균 기간), `ma_long` (장기 이동평균 기간)

2. **RSI 전략 (RSIStrategy)**
   - 상대강도지수(RSI)가 과매수/과매도 수준에서 반등/하락할 때 매매 신호 생성
   - 파라미터: `rsi_period` (RSI 계산 기간), `oversold` (과매도 임계값), `overbought` (과매수 임계값)

3. **MACD 전략 (MACDStrategy)**
   - MACD 라인과 시그널 라인의 교차를 기반으로 매매 신호 생성
   - 파라미터: `macd_fast` (빠른 EMA 기간), `macd_slow` (느린 EMA 기간), `macd_signal` (시그널 라인 기간)

4. **복합 전략 (CompositeStrategy)**
   - 여러 기본 전략의 신호를 가중 평균하여 최종 매매 신호 생성
   - 파라미터: 각 전략의 파라미터와 가중치, 신호 임계값(`threshold`)

### 3.2 백테스터 (Backtester)

백테스터는 다음과 같은 과정으로 전략의 백테스트를 수행합니다:

1. 전략으로부터 매매 신호 생성
2. 매매 신호에 따라 포지션 진입 및 청산
3. 포트폴리오 가치와 수익률 계산
4. 거래 기록 및 성과 지표 계산

```python
class Backtester:
    def __init__(self, data, initial_capital=10000.0, commission=0.001):
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run(self, strategy):
        # 백테스트 실행 로직
        # ...
        return BacktestResult(...)
```

### 3.3 백테스트 결과 (BacktestResult)

백테스트 결과는 다음과 같은 주요 정보를 담고 있습니다:

1. 전략 정보 (이름, 파라미터)
2. 포트폴리오 가치 및 포지션 시계열
3. 거래 이력
4. 성과 지표 (총 수익률, 연간 수익률, 샤프 비율, 최대 낙폭, 승률 등)

또한, 결과를 CSV로 저장하고 다양한 형태의 시각화를 제공합니다:

- 포트폴리오 성과 시각화 (`plot_portfolio_performance`)
- 누적 수익률 비교 차트 (`plot_cumulative_returns`)

### 3.4 파라미터 최적화

그리드 탐색을 통해 최적의 전략 파라미터를 찾는 기능을 제공합니다:

```python
def parameter_grid_search(backtester, strategy_class, param_grid, strategy_name="GridSearch"):
    # 파라미터 조합 생성
    # 모든 조합에 대해 백테스트 실행
    # 최적 결과 반환
    return all_results, best_result
```

## 4. 주요 성과 지표

백테스트 결과는 다음과 같은 주요 성과 지표를 계산합니다:

1. **총 수익률(Total Return)**: 전체 백테스트 기간 동안의 수익률
2. **연간 수익률(Annual Return)**: 연간화된 수익률 (CAGR)
3. **변동성(Volatility)**: 일일 수익률의 표준편차 (연간화)
4. **최대 낙폭(Maximum Drawdown)**: 최대 손실 비율
5. **샤프 비율(Sharpe Ratio)**: 위험 대비 수익 비율
6. **승률(Win Rate)**: 수익 거래의 비율
7. **수익 대비 위험(Return/Risk Ratio)**: 연간 수익률 / 변동성

## 5. 사용 방법

### 5.1 기본 사용법

```bash
# 기본 실행 (샘플 데이터 생성)
python src/run_backtest.py

# 외부 데이터 사용
python src/run_backtest.py --data /path/to/data.csv

# 결과 디렉토리 지정
python src/run_backtest.py --results /custom/results/dir --charts /custom/charts/dir
```

### 5.2 커스텀 전략 구현

새로운 전략을 추가하려면 `Strategy` 클래스를 상속받아 `generate_signals` 메서드를 구현합니다:

```python
class MyCustomStrategy(Strategy):
    def generate_signals(self, data):
        # 매매 신호 생성 로직
        signals = pd.Series(0, index=data.index)
        # ...
        return signals
```

### 5.3 결과 해석

백테스트 결과는 다음과 같은 형태로 해석할 수 있습니다:

1. **CSV 파일**: 모든 전략 및 파라미터 조합의 성과 지표를 담고 있음
2. **성과 차트**: 포트폴리오 가치, 포지션, 낙폭 등의 시각화
3. **누적 수익률 차트**: 전략 간 성과 비교 및 벤치마크 대비 성과 시각화

## 6. 한계점 및 주의사항

1. **과적합(Overfitting) 위험**: 과거 데이터에 최적화된 전략이 미래에도 잘 작동한다는 보장 없음
2. **데이터 편향**: 특정 기간/시장 상황에서만 테스트할 경우 편향된 결과 가능
3. **슬리피지(Slippage) 및 시장 충격**: 실제 시장에서는 주문 실행 시 가격 차이 및 시장 충격 발생 가능
4. **거래 비용**: 모든 거래 비용(수수료, 세금 등)을 정확히 모델링하기 어려움
5. **생존 편향(Survivorship Bias)**: 현존하는 종목만 분석할 경우 발생하는 편향

## 7. 향후 개선 방향

1. **머신러닝 기반 전략 추가**: 예측 모델을 활용한 트레이딩 전략 구현
2. **포트폴리오 최적화**: 여러 자산에 걸친 포트폴리오 구성 및 최적화
3. **워크-포워드 분석(Walk-Forward Analysis)**: 과적합 방지를 위한 방법론 적용
4. **대체 최적화 알고리즘**: 유전 알고리즘, 베이지안 최적화 등의 적용
5. **실시간 백테스팅 모니터링**: 웹 인터페이스를 통한 백테스트 결과 모니터링 도구

## 8. 참고 문헌

1. López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
2. Chan, E. P. (2013). Algorithmic Trading: Winning Strategies and Their Rationale. Wiley.
3. Kakushadze, Z., & Serur, J. A. (2018). 151 Trading Strategies. Palgrave Macmillan.

## 9. 부록: 주요 코드 예시

### 9.1 복합 전략 생성

```python
def create_composite_strategy(name, params):
    # 개별 전략 파라미터 추출
    ma_params = {'ma_short': params.get('ma_short', 10), 'ma_long': params.get('ma_long', 30)}
    rsi_params = {'rsi_period': params.get('rsi_period', 14), 'oversold': params.get('oversold', 30),
                  'overbought': params.get('overbought', 70)}
    macd_params = {'macd_fast': params.get('macd_fast', 12), 'macd_slow': params.get('macd_slow', 26),
                   'macd_signal': params.get('macd_signal', 9)}
    
    # 개별 전략 생성
    ma_strategy = MovingAverageCrossStrategy("MA", ma_params)
    rsi_strategy = RSIStrategy("RSI", rsi_params)
    macd_strategy = MACDStrategy("MACD", macd_params)
    
    # 가중치 설정
    weights = [params.get('weight_ma', 1.0), params.get('weight_rsi', 1.0), params.get('weight_macd', 1.0)]
    
    # 복합 전략 생성
    return CompositeStrategy(name=name, parameters=params,
                            strategies=[ma_strategy, rsi_strategy, macd_strategy],
                            weights=weights)
```

### 9.2 성과 시각화

```python
def plot_portfolio_performance(save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    
    # 가격 및 포트폴리오 가치 그래프
    ax1 = axes[0]
    ax1.plot(data.index, data['close'], 'b-', label='Price')
    ax1_right = ax1.twinx()
    ax1_right.plot(portfolio_value.index, portfolio_value, 'g-', label='Portfolio Value')
    
    # 매수/매도 지점 표시
    ax1.plot(buy_signals, data.loc[buy_signals, 'close'], '^', markersize=10, color='g', label='Buy')
    ax1.plot(sell_signals, data.loc[sell_signals, 'close'], 'v', markersize=10, color='r', label='Sell')
    
    # ... 나머지 시각화 코드 ...
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    return fig
``` 