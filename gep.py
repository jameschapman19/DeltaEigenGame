import numpy as np


class GEPSolver:
    def __init__(self, learning_rate=0.001, epochs=200000, components=1, method="eg"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.components = components
        self.method = method
        self.rho = 1e-3
        self.BU = None

    def fit(self, A, B):
        u = np.random.rand(A.shape[1], self.components)
        u /= np.sqrt(np.diag(u.T @ B @ u))
        for _ in range(self.epochs):
            u = u - self.learning_rate * self.grads(A, B, u)
        return u

    def grads(self, A, B, u):
        Aw = A @ u
        Bw = B @ u
        wAw = u.T @ Aw
        wBw = u.T @ Bw
        if self.method == "delta":
            # grads = 2 * Aw - (Aw @ np.triu(wBw) * np.sign(np.diag(wAw)) + Bw @ np.triu(wAw))
            grads = 2 * Aw - (Aw @ np.triu(wBw) + Bw @ np.triu(wAw))
        elif self.method == "gha":
            grads = 2 * Aw - 2 * Bw @ np.triu(wAw)
        elif self.method == "gamma":
            u /= np.linalg.norm(u, axis=0)
            grads = self.gamma_grads(A, B, u)
        else:
            raise ValueError("Invalid method")
        return -grads

    def gamma_grads(self, A, B, u):
        Aw = A @ u
        Bw = B @ u
        wAw = u.T @ Aw
        wBw = u.T @ Bw
        check = np.diag(wAw) / np.diag(wBw)
        y = u / np.sqrt(np.diag(wBw))
        rewards = Aw * np.diag(wBw) - Bw * np.diag(wAw)
        Ay = A @ y
        By = B @ y
        penalties = np.zeros_like(u)
        penalties += By @ np.triu(Ay.T @ u * np.diag(wBw), 1) - Bw * np.diag(np.tril(u.T @ By, -1) @ Ay.T @ u)
        grads = rewards - penalties
        return grads


def main():
    # A random generalized eigenvalue problem with symetric matrices A and B where A is not positive definite
    A = np.random.rand(10, 10)
    B = np.random.rand(10, 10)
    A = A.T @ A
    U, S, Vt = np.linalg.svd(A)
    # flip the sign of the smallest half of the eigenvalues
    S[5:] = S[:5] * -1
    A = U @ np.diag(S) @ Vt
    B = B.T @ B

    from scipy.linalg import eigh
    # get largest eigenvalue and eigenvector
    w, u_ = eigh(A, B)
    print(np.diag(u_.T @ A @ u_))
    solver = GEPSolver(components=7, method="gamma")
    u_eg = solver.fit(A, B)
    print(np.diag(u_eg.T @ A @ u_eg))
    print(np.diag(u_eg.T @ B @ u_eg))
    solver = GEPSolver(components=7, method="delta")
    u_eg = solver.fit(A, B)
    print(np.diag(u_eg.T @ A @ u_eg))
    print(np.diag(u_eg.T @ B @ u_eg))
    solver = GEPSolver(components=7, method="gha")
    u_gha = solver.fit(A, B)
    print(np.diag(u_gha.T @ A @ u_gha))
    print(np.diag(u_gha.T @ B @ u_gha))


if __name__ == '__main__':
    main()
