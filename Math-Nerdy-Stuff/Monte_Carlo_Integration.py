import numpy as np

def monte_carlo_integration(func, a, b, n_samples=100000):
    x = np.random.uniform(a, b, n_samples)
    y = func(x)
    integral = (b - a) * np.mean(y)
    return integral

result = monte_carlo_integration(lambda x: x**2/2, 0,3)
print(result)
