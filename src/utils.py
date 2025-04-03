"""
공통 유틸리티 함수 모듈
다양한 실험에서 재사용할 수 있는 범용 기능을 제공합니다.
"""

import os
import logging
import json
import pickle
from datetime import datetime
from typing import Dict, List, Union, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """로깅 설정을 위한 헬퍼 함수

    Args:
        name: 로거 이름
        log_file: 로그 파일 경로
        level: 로깅 레벨

    Returns:
        설정된 로거 객체
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 파일 핸들러 생성
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # 콘솔 핸들러 생성
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

    return logger


def ensure_directory(directory_path: str) -> None:
    """경로가 존재하지 않으면 생성합니다

    Args:
        directory_path: 생성할 디렉토리 경로
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


def save_json(data: Dict, file_path: str) -> None:
    """데이터를 JSON 파일로 저장합니다

    Args:
        data: 저장할 데이터 (딕셔너리)
        file_path: 저장할 파일 경로
    """
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file_path: str) -> Dict:
    """JSON 파일을 로드합니다

    Args:
        file_path: 로드할 파일 경로

    Returns:
        로드된 데이터
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, file_path: str) -> None:
    """데이터를 피클 파일로 저장합니다

    Args:
        data: 저장할 데이터
        file_path: 저장할 파일 경로
    """
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path: str) -> Any:
    """피클 파일을 로드합니다

    Args:
        file_path: 로드할 파일 경로

    Returns:
        로드된 데이터
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def timestamp_string() -> str:
    """현재 시간을 포맷팅된 문자열로 반환합니다

    Returns:
        YYYYMMDD_HHMMSS 형식의 타임스탬프
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_metrics(
    metrics: Dict[str, List], title: str, save_path: Optional[str] = None
) -> plt.Figure:
    """학습 메트릭을 시각화합니다

    Args:
        metrics: 메트릭 이름과 값 목록의 딕셔너리
        title: 플롯 제목
        save_path: 저장할 파일 경로 (선택적)

    Returns:
        생성된 Figure 객체
    """
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))

    if len(metrics) == 1:
        axes = [axes]

    for i, (metric_name, values) in enumerate(metrics.items()):
        axes[i].plot(values)
        axes[i].set_title(f"{metric_name}")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric_name)
        axes[i].grid(True)

    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)

    if save_path:
        ensure_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def get_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """실험 결과를 저장할 디렉토리 경로를 생성하고 반환합니다

    Args:
        base_dir: 기본 디렉토리 경로
        experiment_name: 실험 이름

    Returns:
        타임스탬프가 포함된 실험 디렉토리 경로
    """
    timestamp = timestamp_string()
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    ensure_directory(exp_dir)
    return exp_dir
