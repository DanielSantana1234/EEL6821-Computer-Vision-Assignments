"""
V. Symmetric Matrices and Tridiagonalization
Using co-variance matrices, which are symmetric,
Tri-diagonalize the covariance matrix using (1) Householder method, 
(2) Givens Method and 
(3) the Jacobi method. Keep track of the number of iterations required for each method
Verify that values in the diagonal of the final Jacobi matrix are indeed the eigenvalues of the covariance matrix.
Are all the eigenvalues similar for all three methods?
Determine the total processing time for each method, and plot the required processing time 
versus dimension N of the N x N matrix
Check for all three methods if the eigenvalues obtained are the same as those of the 
original covariance matrix, then see if the eigenvectors of the respective tri-diagonalized matrices 
are related to those of the covariance matrix by the relation yi=T-1xi
"""
import numpy as np
import time
import matplotlib.pyplot as plt

def make_cov(n, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T

def householder(A):
    n = A.shape[0]
    T = A.astype(float).copy()
    Q = np.eye(n)
    iters = 0

    for k in range(n - 2):
        x = T[k+1:, k].copy()
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-14:
            continue

        x[0] += np.sign(x[0]) * norm_x
        x /= np.linalg.norm(x)

        T[k+1:, k:] -= 2 * np.outer(x, x @ T[k+1:, k:])
        T[:, k+1:] -= 2 * np.outer(T[:, k+1:] @ x, x)
        Q[:, k+1:] -= 2 * np.outer(Q[:, k+1:] @ x, x)
        iters += 1

    return T, Q, iters

def givens(A):
    n = A.shape[0]
    T = A.astype(float).copy()
    Q = np.eye(n)
    iters = 0

    for k in range(n - 2):
        for i in range(k + 2, n):
            if abs(T[i, k]) < 1e-14:
                continue
            r = np.hypot(T[k+1, k], T[i, k])
            c, s = T[k+1, k] / r, -T[i, k] / r

            G = np.eye(n)
            G[k+1, k+1], G[k+1, i] = c, -s
            G[i, k+1], G[i, i] = s, c

            T = G @ T @ G.T
            Q = Q @ G.T
            iters += 1

    return T, Q, iters

def jacobi(A, tol=1e-10, max_iter=10000):
    n = A.shape[0]
    D = A.astype(float).copy()
    V = np.eye(n)
    iters = 0

    for _ in range(max_iter):
        mask = np.ones((n, n), bool)
        np.fill_diagonal(mask, False)
        p, q = np.unravel_index(np.argmax(np.abs(D) * mask), (n, n))

        if abs(D[p, q]) < tol:
            break

        if abs(D[p, p] - D[q, q]) < 1e-14:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan2(2 * D[p, q], D[p, p] - D[q, q])
        c, s = np.cos(theta), np.sin(theta)

        G = np.eye(n)
        G[p, p], G[p, q] =  c, -s
        G[q, p], G[q, q] =  s,  c

        D = G.T @ D @ G
        V = V @ G
        iters += 1

    return D, V, iters

def check_eigenvector_relation(A, T, Q, name):
    _, X = np.linalg.eigh(A)
    _, Y = np.linalg.eigh(T)

    predicted = np.linalg.inv(Q) @ X

    errs = [min(np.linalg.norm(predicted[:, i] - Y[:, i]),
                np.linalg.norm(predicted[:, i] + Y[:, i]))
            for i in range(A.shape[0])]

    print(f" {name}: max ||yi - T^-1 xi|| = {max(errs):.2e}")

def run_timing(sizes):
    t_hh, t_gv, t_jc = [], [], []

    for n in sizes:
        C = make_cov(n, seed=0)

        t0 = time.perf_counter(); householder(C); t_hh.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); givens(C); t_gv.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); jacobi(C); t_jc.append(time.perf_counter() - t0)

        print(f"  N={n:3d}  Householder={t_hh[-1]:.4f}s  "
              f"Givens={t_gv[-1]:.4f}s  Jacobi={t_jc[-1]:.4f}s")

    return t_hh, t_gv, t_jc

def main():
    N = 6
    C = make_cov(N)
    ref_eigs = np.sort(np.linalg.eigvalsh(C))[::-1]

    print(f"Covariance matrix ({N}x{N})")
    print(f"Reference eigenvalues: {np.round(ref_eigs, 4)}")

    T_hh, Q_hh, i_hh = householder(C)
    eigs_hh = np.sort(np.linalg.eigvalsh(T_hh))[::-1]
    print(f"\nHouseholder  |  iterations: {i_hh}")
    print(f"  Eigenvalues: {np.round(eigs_hh, 4)}")
    print(f"  Match reference: {np.allclose(eigs_hh, ref_eigs)}")
    check_eigenvector_relation(C, T_hh, Q_hh, "Householder")

    T_gv, Q_gv, i_gv = givens(C)
    eigs_gv = np.sort(np.linalg.eigvalsh(T_gv))[::-1]
    print(f"\nGivens  |  iterations: {i_gv}")
    print(f"  Eigenvalues: {np.round(eigs_gv, 4)}")
    print(f"  Match reference: {np.allclose(eigs_gv, ref_eigs)}")
    check_eigenvector_relation(C, T_gv, Q_gv, "Givens")

    D_jac, V_jac, i_jac = jacobi(C)
    eigs_jac = np.sort(np.diag(D_jac))[::-1]
    print(f"\nJacobi  |  iterations: {i_jac}")
    print(f"  Diagonal of D (eigenvalues): {np.round(eigs_jac, 4)}")
    print(f"  Match reference: {np.allclose(eigs_jac, ref_eigs)}")
    print(f"  Diagonal of Jacobi D == eigenvalues: {np.allclose(eigs_jac, ref_eigs)}")

    print("Are all three methods' eigenvalues the same?")
    print(f"  HH == Givens : {np.allclose(eigs_hh, eigs_gv)}")
    print(f"  HH == Jacobi : {np.allclose(eigs_hh, eigs_jac)}")
    print(f"  Givens == Jacobi: {np.allclose(eigs_gv, eigs_jac)}")

    sizes = [5, 10, 20, 40, 60, 80, 100]
    print("Timing experiment:")
    t_hh, t_gv, t_jc = run_timing(sizes)

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, t_hh, "o-",  label="Householder")
    plt.plot(sizes, t_gv, "s--", label="Givens")
    plt.plot(sizes, t_jc, "^:",  label="Jacobi")
    plt.xlabel("Matrix Dimension N")
    plt.ylabel("Time (seconds)")
    plt.title("Processing Time vs Matrix Dimension N")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("processing_time_vs_N.png", dpi=150)
    plt.show()
    print("\nPlot saved → processing_time_vs_N.png")


if __name__ == "__main__":
    main()