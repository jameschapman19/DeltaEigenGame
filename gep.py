import numpy as np


class GEPSolver:
    def __init__(
        self,
        learning_rate=1e-5,
        epochs=200000,
        components=1,
        method="eg",
        return_eigvals=True,
        line_search=False,
        momentum=0.0,
    ):
        self.u = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.components = components
        self.method = method
        self.rho = 1e-1
        self.BU = None
        self.return_eigvals = return_eigvals
        self.line_search = line_search
        self.momentum = momentum

    def fit(self, A, B, u=None):
        if u is None:
            u = np.ones((A.shape[0], self.components))
            u /= np.sqrt(np.diag(u.T @ B @ u))
        else:
            u = u.copy()
        self.velocity = np.zeros_like(u)
        self.track = np.zeros((self.epochs, self.components))
        self.subspace_track = np.zeros(self.epochs)
        for _ in range(self.epochs):
            u += self.momentum * self.velocity
            grads = self.grads(A, B, u)
            if self.line_search:
                self.backtracking_line_search(A, B, u, grads)
            self.velocity = self.momentum * self.velocity - self.learning_rate * grads
            u += self.velocity
            self.track[_] = -self.objective(A, B, u)
        self.u = u
        if self.return_eigvals:
            uAu = u.T @ A @ u
            uBu = u.T @ B @ u
            vals = np.diag(uAu) / np.diag(uBu)
            obj = self.objective(A, B, u)
            grad = self.grads(A, B, u)
            self.eigenvalues = vals
        return self

    def grads(self, A, B, u):
        Aw = A @ u
        Bw = B @ u
        wAw = u.T @ Aw
        wBw = u.T @ Bw
        if self.method == "delta":
            grads = 2 * Aw - (Aw @ np.triu(wBw) + Bw @ np.triu(wAw))
        elif self.method == "deltaso":
            grads = 2 * Aw - (Aw @ wBw + Bw @ wAw)
        elif self.method == "ghaso":
            grads = 2 * Aw - 2 * Bw @ wAw
        elif self.method == "gha":
            grads = 2 * Aw - 2 * Bw @ np.triu(wAw)
        elif self.method == "gamma":
            u /= np.linalg.norm(u, axis=0)
            grads = self.gamma_grads(A, B, u)
        else:
            raise ValueError("Invalid method")
        return -grads

    def objective(self, A, B, u):
        Aw = A @ u
        Bw = B @ u
        wAw = u.T @ Aw
        wBw = u.T @ Bw
        obj = (
            -2 * np.trace(2*wAw-wAw@wBw)
        )
        return obj

    def backtracking_line_search(self, A, B, u, grads):
        while (
            self.objective(A, B, u - self.learning_rate * grads)
            > self.objective(A, B, u)
            - self.rho * self.learning_rate * np.linalg.norm(grads) ** 2
        ):
            self.learning_rate *= 0.9
        return

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
        penalties = By @ np.triu(Ay.T @ u * np.diag(wBw), 1) - Bw * np.diag(
            np.tril(u.T @ By, -1) @ Ay.T @ u
        )
        grads = rewards - penalties
        return grads


def main():
    p = 10
    # A random generalized eigenvalue problem with symetric matrices A and B where A is not positive definite
    A = np.random.rand(p, p)
    B = np.random.rand(p, p)
    A = A.T @ A
    U, S, Vt = np.linalg.svd(A)
    S = np.linspace(1, 1e-3, p)
    A = U @ np.diag(S) @ U.T
    B = B.T @ B
    # replace singular vals with 1
    U, S, Vt = np.linalg.svd(B)
    S = np.linspace(1, 1e-3, p)
    B = U @ np.diag(S) @ U.T
    # B=np.eye(p)
    momentum = 0
    components = 10
    u = np.random.rand(p, components)
    # u /= np.sqrt(np.diag(u.T @ B @ u))

    from scipy.linalg import eigh

    # get largest eigenvalue and eigenvector
    w, u_ = eigh(A, B)
    v_true = np.diag(u_.T @ A @ u_)
    # get largest components in descending order
    idx = np.argsort(v_true)[::-1]
    v_true = v_true[idx[:components]]
    # Delta versions
    delta = GEPSolver(method="delta", momentum=momentum, components=components).fit(
        A, B, u=u
    )
    delta_line = GEPSolver(
        method="delta", momentum=momentum, components=components, line_search=True, learning_rate=1
    ).fit(A, B, u=u)
    # deltaso = GEPSolver(method="deltaso", momentum=momentum, components=components).fit(A, B, u=u)
    # GHAGEP versions
    gha = GEPSolver(method="gha", momentum=momentum, components=components).fit(
        A, B, u=u
    )
    # ghaso = GEPSolver(method="ghaso", momentum=momentum, components=components).fit(A, B, u=u)

    # print eigenvalues
    print("True eigenvalues: ", v_true)
    print("Delta eigenvalues: ", delta.eigenvalues)
    # print("Deltaso eigenvalues: ", deltaso.eigenvalues)
    print("GHA eigenvalues: ", gha.eigenvalues)
    # print("GHASO eigenvalues: ", ghaso.eigenvalues)

    # plot the objective
    import matplotlib.pyplot as plt

    # each method has a track attribute that stores the objective value at each iteration for each component
    # plot this with a label for each component in the legend
    plt.plot(-delta.track)
    # add labels for each component
    plt.legend([i for i in range(components)])
    plt.title("Objective")
    plt.show()
    plt.plot(-delta_line.subspace_track, label="deltals")
    plt.legend()
    plt.title("Subspace objective")
    plt.show()

    print()


if __name__ == "__main__":
    main()
