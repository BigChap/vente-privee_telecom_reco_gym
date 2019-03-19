import numpy as np

def random_points_l2_ball(n, dim):
    """

    Args:
        dim: number of dimensions
        n: number of points

    Returns:
        Points uniformly distributed inside a L2 ball.

    """
    x = np.random.normal(size=(dim, n))
    return np.random.uniform(size=n) ** (1 / dim) * x / np.sqrt(np.sum(x**2, axis=0))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 10000
    x = random_points_l2_ball(n, 2)
    plt.plot(x[0,:], x[1,:], 'o', markersize=2)
    plt.show()