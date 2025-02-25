import numpy as np
import matplotlib.pyplot as plt

def test_crank_nicolson(mu=1.0, a=0.0, M=50, N=100, L=1.0, T=0.1):
    """
    Simple test of Crank-Nicolson for the PDE u_t = mu u_xx + a u.
    """
    h = L / M
    k = T / N
    x = np.linspace(0, L, M+1)
    U = np.sin(np.pi * x)  # initial condition, for instance
    r = mu * k / (h**2)

    A = np.zeros((M-1, M-1))
    B = np.zeros((M-1, M-1))
    
    # Fill matrices for Crank-Nicolson
    for i in range(M-1):
        if i > 0:
            A[i,i-1] = -r/2
            B[i,i-1] = r/2
        A[i,i]   = 1 + r + a*k/2
        B[i,i]   = 1 - r - a*k/2
        if i < M-2:
            A[i,i+1] = -r/2
            B[i,i+1] = r/2

    for n in range(N):
        # Solve A * U_inner^{n+1} = B * U_inner^n
        U_inner = U[1:-1]
        rhs = B @ U_inner
        U_new_inner = np.linalg.solve(A, rhs)
        U = np.concatenate(([0], U_new_inner, [0]))  # Dirichlet BCs

    # Plot the result
    plt.figure()
    plt.plot(x, U, label="CN solution at t=%4.3f" % T)
    plt.title("Crank-Nicolson: mu=%.2f, a=%.2f, M=%d, N=%d" % (mu, a, M, N))
    plt.xlabel("x")
    plt.ylabel("u(x,T)")
    plt.legend()
    plt.show()
    
# Unstable test
test_crank_nicolson(mu=1.0, a=0.0, M=50, N=100, L=1.0, T=0.1)