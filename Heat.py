
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.sparse import diags, eye, csr_matrix
from scipy.sparse.linalg import spsolve
import scienceplots

plt.style.use('science')

class HeatSolver:
    """
    Class for solving the heat equation with linear reaction term using the Crank-Nicolson scheme:
    u_t = μ u_xx + a u
    """
    def __init__(self, mu=1.0, a=0.0, L=1.0, T=1.0):
        self.mu = mu  # Diffusion coefficient
        self.a = a    # Reaction coefficient
        self.L = L    # Domain length
        self.T = T    # Final time
    
    def exact_solution(self, x, t):
        return np.exp((self.a - self.mu * (np.pi**2)) * t) * np.sin(np.pi * x)
        
    def solve_crank_nicolson(self, M, N, return_all=False):
        h = self.L / M  # Spatial step
        k = self.T / N  # Time step 
        r = self.mu * k / (h**2)
        
        # Grid points
        x = np.linspace(0, self.L, M+1)
        t = np.linspace(0, self.T, N+1)
        
        # Solution
        U = np.zeros((N+1, M+1))
        U[0, :] = self.exact_solution(x, 0)  # Initial condition
        
        main_diag = np.ones(M-1) * (1 + r)
        off_diag = np.ones(M-2) * (-r/2)
        
        A = diags(
            [off_diag, main_diag, off_diag],
            [-1, 0, 1],
            shape=(M-1, M-1)
        )
        
      
        for n in range(N):
            t_n = n * k 
            
            # Set up right-hand side for the predictor step (interior points only)
            if return_all:
                U_n = U[n, :]
                b = np.zeros(M-1)
                for i in range(1, M):
                    b[i-1] = (1 - r) * U_n[i] + (r/2) * (U_n[i+1] + U_n[i-1]) + k * self.a * U_n[i]
            else:
                b = np.zeros(M-1)
                for i in range(1, M):
                    b[i-1] = (1 - r) * U[i] + (r/2) * (U[i+1] + U[i-1]) + k * self.a * U[i]
            
            # Solve the predictor step for U^* (interior points): A U^* = b
            U_star_interior = spsolve(A, b)
            
            # Create full U^* array including boundary points
            U_star = np.zeros(M+1)
            U_star[1:M] = U_star_interior
            U_star[0] = 0  # Boundary condition at x=0
            U_star[M] = 0  # Boundary condition at x=L
            
            # Apply corrector step: U^(n+1) = U^* + (k/2)a(U^* - U^n)
            if return_all:
                U[n+1, 0] = 0  # Left boundary
                U[n+1, M] = 0  # Right boundary
                for i in range(1, M):
                    U[n+1, i] = U_star[i] + (k/2) * self.a * (U_star[i] - U_n[i])
            else:
                U_new[0] = 0  # Left boundary
                U_new[M] = 0  # Right boundary
                for i in range(1, M):
                    U_new[i] = U_star[i] + (k/2) * self.a * (U_star[i] - U[i])
                U = np.copy(U_new)
        
        return (U, x, t) if return_all else (U, x)
    
    def compute_error(self, M, N):
        """Compute the maximum error between the numerical and exact solutions."""
        U_numerical, x = self.solve_crank_nicolson(M, N)
        U_exact = self.exact_solution(x, self.T)
        return np.max(np.abs(U_numerical - U_exact))
    
    def convergence_analysis(self, M_values, parabolic_scaling=True):
        """
        Perform convergence analysis for different spatial resolutions.
        If parabolic_scaling is True, use N ~ M2 (k ~ h2).
        """
        errors = []
        h_values = [self.L / M for M in M_values]
        
        for M in M_values:
            if parabolic_scaling:
                N = M**2 
            else:
                N = 1000
            
            error = self.compute_error(M, N)
            errors.append(error)
            print(f"M = {M}, N = {N}, h = {self.L/M:.6f}, k = {self.T/N:.6f}, max error = {error:.6e}")
        
        return h_values, errors

    def local_truncation_error(self, M_values, parabolic_scaling=True):
        """
        Estimate local truncation error by computing the difference between the 
        true derivative and the numerical approximation at t = k.
        """
        lte_values = []
        h_values = [self.L / M for M in M_values]
        
        for M in M_values:
            if parabolic_scaling:
                N = M**2 // 4
            else:
                N = 1000
            
            h = self.L / M
            k = self.T / N
            
            # One step from initial condition
            U_numerical, x, t = self.solve_crank_nicolson(M, 1, return_all=True)
            
            # Exact solution at t = k
            U_exact_1 = self.exact_solution(x, k)
            
            # Exact solution at t = 0
            U_exact_0 = self.exact_solution(x, 0)
            
            # True derivative at t = 0
            true_derivative = (U_exact_1 - U_exact_0) / k
            
            # Numerical derivative
            numerical_derivative = (U_numerical[1, :] - U_numerical[0, :]) / k
            
            # Local truncation error is the difference
            lte = np.max(np.abs(numerical_derivative - true_derivative))
            lte_values.append(lte)
            
            print(f"M = {M}, N = {N}, LTE = {lte:.6e}")
        
        return h_values, lte_values

    def stability_analysis(self):
        """
        Test the stability of the Crank-Nicolson scheme for different values of r = mu k/h2
        """
        r_values = np.logspace(-1, 3, 10)  # Various mesh ratios
        M = 50  # Fixed spatial resolution
        h = self.L / M
        
        max_errors = []
        
        for r in r_values:
            k = r * h**2 / self.mu
            N = max(1, int(self.T / k))
            
            U_numerical, x = self.solve_crank_nicolson(M, N)
            U_exact = self.exact_solution(x, self.T)
            error = np.max(np.abs(U_numerical - U_exact))
            
            max_errors.append(error)
            print(f"r = {r:.2e}, k/h^2 = {k/h**2:.2e}, max error = {error:.6e}")
        
        return r_values, max_errors

def plot_convergence(h_values, errors, title, expected_order=2):
    """Plot convergence results with a reference line for expected order."""
    plt.figure(figsize=(8, 6), dpi=300)
    
    plt.loglog(h_values, errors, 'o-', label='Numerical Error')
    
    # Add reference line for expected order
    ref_h = np.array([min(h_values), max(h_values)])
    ref_error = errors[0] * (ref_h / h_values[0])**expected_order
    plt.loglog(ref_h, ref_error, 'k--', label=f'$O(h^{expected_order})$', lw=2)
    
    # Calculate observed convergence order
    observed_order = np.log(errors[-2]/errors[-1]) / np.log(h_values[-2]/h_values[-1])
    
    plt.title(f"{title}\nObserved Order: {observed_order:.2f}")
    plt.xlabel('Step Size (h)')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()
    
    return observed_order

def plot_solution_evolution(U, x, t, exact_solution=None, mu=1.0, a=0.0, title="Evolution of Solution"):
    """Plot the solution evolution over time."""
    fig = plt.figure(figsize=(12, 8))
    
    # 3D surface plot
    ax1 = fig.add_subplot(221, projection='3d')
    X, T = np.meshgrid(x, t)
    surf = ax1.plot_surface(X, T, U, cmap=cm.viridis, linewidth=0, antialiased=True)
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('Time (t)')
    ax1.set_zlabel('Temperature (u)')
    ax1.set_title('Solution Surface')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Contour plot
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(X, T, U, 50, cmap=cm.viridis)
    ax2.set_xlabel('Position (x)')
    ax2.set_ylabel('Time (t)')
    ax2.set_title('Solution Contour')
    fig.colorbar(contour, ax=ax2)
    
    # Select a few time points to plot
    time_indices = np.linspace(0, len(t)-1, 6, dtype=int)
    ax3 = fig.add_subplot(212)
    
    for idx in time_indices:
        current_t = t[idx]
        ax3.plot(x, U[idx], label=f't = {current_t:.2f}')
        
        # Plot exact solution if provided
        if exact_solution is not None:
            u_exact = np.array([exact_solution(xi, current_t) for xi in x])
            ax3.plot(x, u_exact, 'k--', alpha=0.5)
    
    ax3.set_xlabel('Position (x)')
    ax3.set_ylabel('Temperature (u)')
    ax3.set_title('Solution at Different Times')
    ax3.legend()
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    return fig

def run_validation_tests():
    """Run validation tests to support the theoretical results."""
    print("\n========== VALIDATION TESTS FOR HEAT EQUATION WITH CRANK-NICOLSON ==========")
    
    # Test case parameters
    mu = 1.0  # Diffusion coefficient
    a = 2.0   # Reaction coefficient (try both positive and negative values)
    
    # Create solver
    solver = HeatSolver(mu=mu, a=a)
    
    # === 1. Convergence Analysis with parabolic scaling (k ~ h2) ===
    print("\n1. CONVERGENCE ANALYSIS (parabolic scaling: k ~ h2)")
    M_values = [10, 20, 40, 80, 160]
    h_values, errors = solver.convergence_analysis(M_values, parabolic_scaling=True)
    order = plot_convergence(h_values, errors, "Convergence with Parabolic Scaling (k ~ h2)")
    plt.savefig('./figures/convergence_parabolic_scaling.png', dpi=300, bbox_inches='tight')
    
    # === 2. Local Truncation Error Analysis ===
    print("\n2. LOCAL TRUNCATION ERROR ANALYSIS")
    h_values, lte_values = solver.local_truncation_error(M_values, parabolic_scaling=True)
    order = plot_convergence(h_values, lte_values, "Local Truncation Error with Parabolic Scaling")
    plt.savefig('./figures/local_truncation_error.png', dpi=300, bbox_inches='tight')
    
    # === 3. Stability Analysis ===
    print("\n3. STABILITY ANALYSIS FOR DIFFERENT VALUES OF r = mu*k/h2")
    r_values, max_errors = solver.stability_analysis()
    
    plt.figure()
    plt.semilogx(r_values, max_errors, 'o-')
    plt.xlabel('$r = \frac{\mu k}{h^2}$')
    plt.ylabel('Maximum Error')
    plt.title('Stability Analysis for Different Values of r')
    plt.grid(True)
    plt.savefig('./figures/stability_analysis.png', dpi=300, bbox_inches='tight')
    
    # === 4. Visual Validation with Exact Solution ===
    print("\n4. VISUAL VALIDATION WITH EXACT SOLUTION")
    M, N = 50, 100
    U, x, t = solver.solve_crank_nicolson(M, N, return_all=True)
    
    fig = plot_solution_evolution(U, x, t, exact_solution=solver.exact_solution, mu=mu, a=a,
                                 title=f"Heat Equation Solution $(\mu={mu}, a={a})$")
    plt.savefig('./figures/solution_evolution.png', dpi=300, bbox_inches='tight')
    
    # === 5. Error over time ===
    print("\n5. ERROR EVOLUTION OVER TIME")
    max_errors_over_time = []
    
    for n in range(len(t)):
        U_exact = np.array([solver.exact_solution(xi, t[n]) for xi in x])
        error = np.max(np.abs(U[n] - U_exact))
        max_errors_over_time.append(error)
    
    plt.figure()
    plt.semilogy(t, max_errors_over_time)
    plt.xlabel('Time (t)')
    plt.ylabel('Maximum Error')
    plt.title('Error Evolution Over Time')
    plt.grid(True)
    plt.savefig('./figures/error_over_time.png', dpi=300, bbox_inches='tight')
    
    # === 6. Effect of Reaction Term ===
    print("\n6. EFFECT OF REACTION TERM")
    a_values = [-5.0, -1.0, 0.0, 1.0, 5.0]
    plt.figure(figsize=(12, 6))
    
    for a_val in a_values:
        solver_a = HeatSolver(mu=mu, a=a_val)
        U, x, t = solver_a.solve_crank_nicolson(M=50, N=100, return_all=True)
        
        # Plot final solution
        plt.plot(x, U[-1], label=f'a = {a_val}')
        
        # Plot exact solution
        U_exact = np.array([solver_a.exact_solution(xi, t[-1]) for xi in x])
        plt.plot(x, U_exact, 'k--', alpha=0.5)
    
    plt.xlabel('Position (x)')
    plt.ylabel('Temperature (u)')
    plt.title(f'Effect of Reaction Term a at t = {t[-1]}')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figures/effect_of_reaction_term.png', dpi=300, bbox_inches='tight')
    
    
    print("\n========== VALIDATION COMPLETE ==========")
    print("All figures saved to disk.")


run_validation_tests()
plt.show()