# 피보나치 수열 알고리즘 비교 연구

**작성일**: 2023-04-03
**관련 실험**: exp_01_fibonacci

## 개요
이 문서는 다양한 피보나치 수열 계산 알고리즘의 구현과 성능 비교에 대한 공식 보고서입니다. 피보나치 수열 계산은 알고리즘 효율성을 비교하는 좋은 예시로, 다양한 접근 방식에 따른 시간 복잡도와 공간 복잡도의 차이를 보여줍니다.

## 알고리즘 구현
총 5가지 방식으로 피보나치 수열 알고리즘을 구현했습니다:

### 1. 기본 반복문 방식
```python
def fibonacci(n):
    fib_sequence = [0, 1]
    
    for i in range(2, n):
        fib_sequence.append(fib_sequence[i - 1] + fib_sequence[i - 2])
    
    return fib_sequence
```
- **시간 복잡도**: O(n)
- **공간 복잡도**: O(n)

### 2. 재귀 호출 방식
```python
def fibonacci_recursive(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
```
- **시간 복잡도**: O(2^n)
- **공간 복잡도**: O(n) (호출 스택)

### 3. 동적 프로그래밍 방식
```python
def fibonacci_dynamic(n):
    memo = [0] * n
    if n > 0: memo[0] = 0
    if n > 1: memo[1] = 1
    
    for i in range(2, n):
        memo[i] = memo[i - 1] + memo[i - 2]
    
    return memo
```
- **시간 복잡도**: O(n)
- **공간 복잡도**: O(n)

### 4. 제너레이터 방식
```python
def fibonacci_generator(n):
    a, b = 0, 1
    count = 0
    
    while count < n:
        yield a
        a, b = b, a + b
        count += 1
```
- **시간 복잡도**: O(n)
- **공간 복잡도**: O(1) (각 단계마다 일정 공간)

### 5. 행렬 거듭제곱 방식
```python
def fibonacci_matrix(n):
    # 행렬 거듭제곱을 이용한 피보나치 계산
    # 구현 생략 (전체 코드는 src/fibonacci.py 참조)
```
- **시간 복잡도**: O(log n)
- **공간 복잡도**: O(1)

## 성능 비교 결과

| 알고리즘 | n=10 | n=20 | n=30 | n=40 |
|---------|------|------|------|------|
| 반복 | < 0.001s | < 0.001s | < 0.001s | < 0.001s |
| 동적 프로그래밍 | < 0.001s | < 0.001s | < 0.001s | < 0.001s |
| 제너레이터 | < 0.001s | < 0.001s | < 0.001s | < 0.001s |
| 행렬 거듭제곱 | < 0.001s | < 0.001s | < 0.001s | < 0.001s |
| 재귀 | < 0.001s | 0.005s | 0.32s | 초과 |

자세한 벤치마크 결과는 `fibonacci_benchmark.png` 파일을 참조하세요.

## 결론

1. **작은 n 값**에서는 모든 알고리즘이 비슷한 성능을 보입니다.

2. **중간 n 값**에서는 재귀 알고리즘의 성능이 급격히 저하되기 시작합니다.

3. **큰 n 값**에서는 행렬 거듭제곱 방식이 가장 효율적입니다.

4. 일반적인 사용에서 **메모리 효율성**을 중요시한다면 제너레이터 방식을 권장합니다.

5. **계산 효율성**을 최우선으로 한다면 행렬 거듭제곱 방식을 사용해야 합니다.

## 향후 연구 방향

1. 병렬 처리를 이용한 대규모 피보나치 수 계산 최적화
2. 더 큰 피보나치 수를 위한 BigInteger 구현
3. 행렬 거듭제곱 알고리즘 최적화 방안 연구

## 참고 자료

- [피보나치 수열의 수학적 특성](https://en.wikipedia.org/wiki/Fibonacci_number)
- [행렬 거듭제곱을 이용한 선형 점화식 계산](https://en.wikipedia.org/wiki/Exponentiation_by_squaring)

## Lesson Learned

- 알고리즘 설계 시 시간 복잡도와 공간 복잡도 모두 고려 필요
- 동일한 문제에 대해 다양한 접근 방식이 존재하며, 각각 상황에 맞게 선택 필요
- 수학적 특성(행렬 관계)을 활용한 알고리즘이 단순 구현보다 효율적일 수 있음
- 처리해야 할 데이터 규모에 따라 적절한 알고리즘 선택이 중요 