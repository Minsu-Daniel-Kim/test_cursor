# 피보나치 시퀀스 구현 문서

## 목적
- 여러 가지 방식으로 피보나치 수열을 구현하여 각 접근법의 특성을 비교
- 재귀, 동적 프로그래밍, 반복, 생성자 기반 접근법의 장단점 분석
- 개발자가 다양한 알고리즘 패턴을 이해할 수 있도록 도움

## 구현된 기능
1. **반복적 접근법 (Iterative)**
   - 목적: 간단하고 직관적인 방식으로 피보나치 수열 생성
   - 특성: 리스트를 사용하여 수열을 저장하고 순차적으로 계산

2. **재귀적 접근법 (Recursive)**
   - 목적: 피보나치의 수학적 정의를 직접 구현하여 이해도 향상
   - 특성: 큰 숫자에 대해 성능이 저하되는 단점 존재

3. **동적 프로그래밍 접근법 (Dynamic Programming)**
   - 목적: 메모이제이션을 활용한 성능 최적화
   - 특성: 중복 계산을 방지하여 효율성 증가

4. **생성자 기반 접근법 (Generator)**
   - 목적: 메모리 효율성 향상
   - 특성: 피보나치 수를 한 번에 하나씩 생성하여 메모리 사용 최소화

## 실행 방법

### 코드 실행
```python
python fibonacci.py
```

### 사용 예시
1. 프로그램 실행 시 생성할 피보나치 수열의 길이 입력:
```
Enter how many Fibonacci numbers to generate: 10
```

2. 구현 방식 선택:
```
Select implementation:
1. Standard iterative approach
2. Dynamic programming approach
3. Generator-based approach
4. Recursive approach (not recommended for large n)

Enter your choice (1-4): 1
```

3. 결과 예시:
```
Fibonacci sequence (iterative):
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

## 성능 고려사항
- 재귀 방식은 n > 30일 경우 매우 느려질 수 있음
- 동적 프로그래밍과 반복 방식은 대부분의 사용 사례에 적합
- 생성자 방식은 메모리 효율성이 중요할 때 유용 