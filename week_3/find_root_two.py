from math import sqrt


def f(x) -> float:
    """
    Function to minimise.
    Has 2 minima where x = ±2^(1/2)
    """
    return (x**2 - 2)**2


def grad_f(x) -> float:
    """
    Returns gradient of f(x)
    f'(x) = 4x^3 - 8x
    """
    return 4 * x**3 - 8 * x


def grad_descent(
    x: float = 2,
    alpha: float = 0.01,
    steps: int = 100,
    early_stopping: bool = False,
    tolerance: float = 1e-10
) -> list[float]:
    stop_early = False
    x_history = [x]

    for i in range(steps):
        if stop_early:
            break

        gradient = grad_f(x)

        if early_stopping and abs(gradient) < tolerance:
            print(f"Converged at iteration {i}, gradient = {gradient}")
            break

        x -= alpha * gradient

        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}: x = {x:.10f}, f(x) = {f(x):.2e}")

        x_history.append(x)

    return x_history


def evaluate_results(x_history: list):
    last_x = str(x_history[-1])
    print("\n\n")
    print("[Result Evaluation]\n")
    print(f"Actual value of 2^(1/2):    {sqrt(2):.15}")
    print(f"Final estimate:             {last_x:.15}")


def main():
    x_history = grad_descent(x=2, alpha=0.01, steps=1000, early_stopping=True)
    evaluate_results(x_history)


if __name__ == "__main__":
    main()
