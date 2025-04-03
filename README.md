# Project AIMO

## 개요
이 저장소는 알고리즘 및 트레이딩 전략 개발을 위한 프로젝트입니다.

## 디렉토리 구조

1. **docs/**: 모든 마크다운 형식 보고서 및 문서
2. **experiments/**: 실험별 코드와 결과물
   - **exp_01_baseline/**: 기준 모델 실험
   - **exp_02_backtesting/**: 백테스팅 관련 실험
3. **data/**: 원본 및 중간/전처리 데이터
4. **src/**: 재사용 가능한 코드와 유틸리티 함수
5. **scripts/**: 자동화 스크립트 및 실행 파일
6. **models/**: 학습된 모델 체크포인트

## 프로젝트 설정

### 의존성 설치
```bash
pip install -r requirements.txt
```

## 실험 실행 방법

각 실험은 해당 실험 폴더 내에서 실행합니다:

```bash
cd experiments/exp_02_backtesting
python src/run_backtest.py
```

## Git LFS 사용 안내
50MB 이상의 파일(대형 CSV, 모델 체크포인트 등)은 반드시 LFS로 관리합니다. 