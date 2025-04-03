import time
import matplotlib.pyplot as plt
from fibonacci import (
    fibonacci,
    fibonacci_dynamic,
    fibonacci_generator,
    fibonacci_recursive,
    fibonacci_matrix,
)


def benchmark(func, n, name=None, max_time=5):
    """
    벤치마크 함수: 주어진 함수의 실행 시간을 측정합니다.
    max_time 초를 초과하면 중단합니다.
    """
    start_time = time.time()

    if name is None:
        name = func.__name__

    try:
        if func == fibonacci_generator:
            result = list(func(n))
        elif func == fibonacci_recursive:
            # 재귀 함수의 경우 각 피보나치 수를 개별적으로 계산
            result = [func(i) for i in range(n)]
        else:
            result = func(n)

        end_time = time.time()
        execution_time = end_time - start_time

        if execution_time > max_time:
            print(f"{name}: 실행 시간 초과 ({execution_time:.6f}초)")
            return execution_time, None

        return execution_time, result

    except Exception as e:
        print(f"{name} 오류: {e}")
        return None, None


def run_benchmarks(n_values):
    """
    다양한 n 값에 대해 모든 알고리즘을 벤치마크합니다.
    """
    algorithms = [
        (fibonacci, "반복"),
        (fibonacci_dynamic, "동적 프로그래밍"),
        (fibonacci_generator, "제너레이터"),
        (fibonacci_matrix, "행렬 거듭제곱"),
    ]

    # n이 작을 때만 재귀 알고리즘 추가
    if max(n_values) <= 30:
        algorithms.append((fibonacci_recursive, "재귀"))

    results = {name: [] for _, name in algorithms}

    for n in n_values:
        print(f"\n== n={n} 벤치마크 ==")

        for func, name in algorithms:
            if name == "재귀" and n > 30:
                results[name].append(None)
                continue

            time_taken, _ = benchmark(func, n, name)
            print(f"{name}: {time_taken:.6f}초")
            results[name].append(time_taken)

    return results


def plot_results(n_values, results):
    """
    벤치마크 결과를 그래프로 시각화합니다.
    """
    plt.figure(figsize=(10, 6))

    for name, times in results.items():
        # None 값은 건너뛰고 그래프 그리기
        valid_points = [(n, t) for n, t in zip(n_values, times) if t is not None]
        if valid_points:
            x, y = zip(*valid_points)
            plt.plot(x, y, marker="o", label=name)

    plt.xlabel("n (피보나치 수열 길이)")
    plt.ylabel("실행 시간 (초)")
    plt.title("피보나치 알고리즘 성능 비교")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 그래프 저장
    plt.savefig("../fibonacci_benchmark.png")
    print("벤치마크 결과가 fibonacci_benchmark.png 파일로 저장되었습니다.")

    # 그래프 표시
    plt.show()


def main():
    """
    메인 벤치마크 함수
    """
    print("피보나치 알고리즘 벤치마크 도구")
    print("-------------------------------")

    # 기본 n 값 설정
    n_values = [5, 10, 15, 20, 25, 30, 35, 40]

    try:
        # 사용자 정의 n 값
        custom = input("사용자 정의 n 값을 사용하시겠습니까? (y/n): ")
        if custom.lower() == "y":
            input_values = input("n 값을 공백으로 구분하여 입력하세요: ")
            n_values = [int(x) for x in input_values.split()]
    except ValueError:
        print("유효하지 않은 입력입니다. 기본값을 사용합니다.")

    print(f"벤치마크 n 값: {n_values}")

    # 재귀 함수 경고
    if max(n_values) > 30:
        print(
            "\n경고: n > 30인 경우 재귀 알고리즘은 매우 느립니다. 벤치마크에서 제외됩니다."
        )

    # 벤치마크 실행
    results = run_benchmarks(n_values)

    # 결과 시각화
    try:
        plot_results(n_values, results)
    except ImportError:
        print(
            "matplotlib 라이브러리가 설치되지 않았습니다. 그래프를 표시할 수 없습니다."
        )
        print("pip install matplotlib 명령으로 설치할 수 있습니다.")


if __name__ == "__main__":
    main()
