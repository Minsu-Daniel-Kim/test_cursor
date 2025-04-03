"""
실험 템플릿 파일

새로운 실험을 시작할 때 이 템플릿을 복사하여 시작점으로 사용할 수 있습니다.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

# 상위 디렉토리를 시스템 경로에 추가하여 src 모듈을 임포트할 수 있도록 합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils import (
    setup_logger,
    ensure_directory,
    save_json,
    plot_metrics,
    get_experiment_dir,
)


class Experiment:
    """실험 기본 클래스

    새로운 실험을 위한 기본 구조를 제공합니다.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 실험 설정을 담고 있는 딕셔너리
        """
        self.config = config
        self.experiment_name = config.get("experiment_name", "experiment")
        self.results_dir = config.get("results_dir", "../results")

        # 실험 디렉토리 설정
        self.experiment_dir = get_experiment_dir(self.results_dir, self.experiment_name)

        # 로거 설정
        log_file = os.path.join(self.experiment_dir, "experiment.log")
        self.logger = setup_logger(self.experiment_name, log_file)

        self.logger.info(f"실험 초기화: {self.experiment_name}")
        self.logger.info(f"결과 저장 경로: {self.experiment_dir}")

        # 설정 저장
        config_path = os.path.join(self.experiment_dir, "config.json")
        save_json(self.config, config_path)
        self.logger.info(f"설정 저장 완료: {config_path}")

    def load_data(self):
        """데이터 로드 메서드

        실험에 필요한 데이터를 로드합니다.
        """
        self.logger.info("데이터 로드 중...")
        # TODO: 데이터 로드 로직 구현
        self.logger.info("데이터 로드 완료")

    def preprocess_data(self):
        """데이터 전처리 메서드

        로드된 데이터를 전처리합니다.
        """
        self.logger.info("데이터 전처리 중...")
        # TODO: 데이터 전처리 로직 구현
        self.logger.info("데이터 전처리 완료")

    def build_model(self):
        """모델 구축 메서드

        실험에 사용할 모델을 구축합니다.
        """
        self.logger.info("모델 구축 중...")
        # TODO: 모델 구축 로직 구현
        self.logger.info("모델 구축 완료")

    def train_model(self):
        """모델 학습 메서드

        구축된 모델을 학습합니다.
        """
        self.logger.info("모델 학습 중...")
        # TODO: 모델 학습 로직 구현

        # 예시: 학습 결과 시각화
        metrics = {
            "loss": [0.5, 0.4, 0.3, 0.2, 0.1],
            "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9],
        }
        plot_metrics(
            metrics,
            f"{self.experiment_name} Training Metrics",
            os.path.join(self.experiment_dir, "training_metrics.png"),
        )

        self.logger.info("모델 학습 완료")

    def evaluate_model(self):
        """모델 평가 메서드

        학습된 모델을 평가합니다.
        """
        self.logger.info("모델 평가 중...")
        # TODO: 모델 평가 로직 구현

        # 예시: 평가 결과 저장
        evaluation_results = {
            "accuracy": 0.92,
            "precision": 0.91,
            "recall": 0.90,
            "f1_score": 0.905,
        }

        save_json(
            evaluation_results,
            os.path.join(self.experiment_dir, "evaluation_results.json"),
        )

        self.logger.info("모델 평가 완료")

    def save_model(self):
        """모델 저장 메서드

        학습된 모델을 저장합니다.
        """
        self.logger.info("모델 저장 중...")
        # TODO: 모델 저장 로직 구현
        self.logger.info("모델 저장 완료")

    def run(self):
        """실험 실행 메서드

        전체 실험 파이프라인을 실행합니다.
        """
        self.logger.info(f"실험 시작: {self.experiment_name}")

        try:
            self.load_data()
            self.preprocess_data()
            self.build_model()
            self.train_model()
            self.evaluate_model()
            self.save_model()

            self.logger.info("실험 성공적으로 완료")
            return True
        except Exception as e:
            self.logger.error(f"실험 중 오류 발생: {str(e)}", exc_info=True)
            return False


def parse_args():
    """명령줄 인자 파싱 함수"""
    parser = argparse.ArgumentParser(description="실험 실행 스크립트")
    parser.add_argument(
        "--config", type=str, default="config.json", help="설정 파일 경로"
    )
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()

    # 설정 파일 로드
    try:
        from src.utils import load_json

        config = load_json(args.config)
    except Exception as e:
        print(f"설정 파일 로드 실패: {str(e)}")
        # 기본 설정 사용
        config = {
            "experiment_name": "baseline_experiment",
            "results_dir": "../results",
            "model_params": {"learning_rate": 0.001, "batch_size": 32, "epochs": 10},
        }

    # 실험 실행
    experiment = Experiment(config)
    success = experiment.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
