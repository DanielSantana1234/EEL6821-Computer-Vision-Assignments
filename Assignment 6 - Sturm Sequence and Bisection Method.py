"""
VI. Sturm Sequence and Bisection Method
Using the Sturm sequence, show how you can determine for a given interval, how many 
eigenvalues reside in that interval? This program should be user interactive.
Using the bisection method, determine to a prescribed accuracy one of the eigenvalues.
"""
import numpy as np

def sturm_count(diag, off, x):
    n = len(diag)
    sign_changes = 0
    p_prev = 1.0
    p_curr = diag[0] - x

    if p_curr < 0:
        sign_changes += 1

    for k in range(1, n):
        p_next = (diag[k] - x) * p_curr - off[k-1]**2 * p_prev
        if (p_next < 0 and p_curr >= 0) or (p_next >= 0 and p_curr < 0):
            sign_changes += 1
        p_prev = p_curr
        p_curr = p_next

    return sign_changes


def count_eigenvalues(diag, off, a, b):
    return sturm_count(diag, off, b) - sturm_count(diag, off, a)


def bisection(diag, off, a, b, tol):
    for i in range(1000):
        mid = (a + b) / 2.0
        if count_eigenvalues(diag, off, a, mid) >= 1:
            b = mid
        else:
            a = mid
        if (b - a) < tol:
            print(f"  Converged in {i+1} iterations.")
            return (a + b) / 2.0
    return (a + b) / 2.0

n = int(input("Enter matrix dimension n: "))

print(f"\nEnter the {n} main diagonal elements:")
diag = [float(input(f"  a[{i+1}] = ")) for i in range(n)]

print(f"\nEnter the {n-1} off diagonal elements:")
off = [float(input(f"  b[{i+1}] = ")) for i in range(1, n)]

T = np.diag(diag)
for i in range(n - 1):
    T[i, i+1] = T[i+1, i] = off[i]
true_eigs = sorted(np.linalg.eigvalsh(T))
print(f"\nReference eigenvalues: {[round(e, 6) for e in true_eigs]}")

while True:
    a = float(input("\nEnter the left  endpoint a: "))
    b = float(input("Enter the right endpoint b: "))
    count = count_eigenvalues(diag, off, a, b)
    print(f" Number of eigenvalues in ({a}, {b}) = {count}")
    if input("\nAnother interval? (y/n): ").lower() != 'y':
        break

while True:
    a = float(input("\nEnter left  endpoint a: "))
    b = float(input("Enter right endpoint b: "))
    tol = float(input("Enter tolerance (e.g. 1e-8): "))

    count = count_eigenvalues(diag, off, a, b)
    if count == 0:
        print("  No eigenvalue in that interval. Try again.")
    elif count > 1:
        print(f"  {count} eigenvalues found narrow the interval to isolate one")
    else:
        result = bisection(diag, off, a, b, tol)
        print(f"  --> Eigenvalue ≈ {result:.10f}  (tolerance = {tol})")

    if input("\nFind another eigenvalue? (y/n): ").lower() != 'y':
        break 