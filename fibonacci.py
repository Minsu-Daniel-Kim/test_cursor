def fibonacci(n):
    """Generate the Fibonacci sequence up to the nth number."""
    fib_sequence = [0, 1]

    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return fib_sequence

    for i in range(2, n):
        fib_sequence.append(fib_sequence[i - 1] + fib_sequence[i - 2])

    return fib_sequence


def fibonacci_recursive(n):
    """Calculate the nth Fibonacci number using recursion.
    Note: This is inefficient for large numbers due to redundant calculations."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_dynamic(n):
    """Generate the Fibonacci sequence up to the nth number using dynamic programming."""
    if n <= 0:
        return []

    # Initialize memo array
    memo = [0] * n

    # Base cases
    if n > 0:
        memo[0] = 0
    if n > 1:
        memo[1] = 1

    # Fill memo array
    for i in range(2, n):
        memo[i] = memo[i - 1] + memo[i - 2]

    return memo


def fibonacci_generator(n):
    """Generate Fibonacci sequence using a generator function."""
    a, b = 0, 1
    count = 0

    while count < n:
        yield a
        a, b = b, a + b
        count += 1


def main():
    try:
        n = int(input("Enter how many Fibonacci numbers to generate: "))
        if n < 0:
            print("Please enter a positive number.")
            return

        print(f"\nSelect implementation:")
        print("1. Standard iterative approach")
        print("2. Dynamic programming approach")
        print("3. Generator-based approach")
        print("4. Recursive approach (not recommended for large n)")

        choice = int(input("\nEnter your choice (1-4): "))

        if choice == 1:
            result = fibonacci(n)
            print(f"\nFibonacci sequence (iterative):")
            print(result)
        elif choice == 2:
            result = fibonacci_dynamic(n)
            print(f"\nFibonacci sequence (dynamic programming):")
            print(result)
        elif choice == 3:
            result = list(fibonacci_generator(n))
            print(f"\nFibonacci sequence (generator):")
            print(result)
        elif choice == 4:
            if n > 30:
                print("\nWarning: Recursive approach may be very slow for n > 30.")
                confirm = input("Continue anyway? (y/n): ")
                if confirm.lower() != "y":
                    return

            result = [fibonacci_recursive(i) for i in range(n)]
            print(f"\nFibonacci sequence (recursive):")
            print(result)
        else:
            print("\nInvalid choice. Please select 1-4.")

    except ValueError:
        print("Please enter a valid integer.")


if __name__ == "__main__":
    main()
