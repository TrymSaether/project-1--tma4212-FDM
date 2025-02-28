import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.animation import FuncAnimation
import scienceplots

from theory import Heat

plt.style.use('science')

def u_exact(x, y, t, mu=1.0, a=0.0):
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-(2 * np.pi**2 * mu - a) * t)


def g_exact(x, y, t, mu=1.0, a=0.0):
    return u_exact(x, y, t, mu, a)


def u0_exact(x, y, mu=1.0, a=0.0):
    return u_exact(x, y, 0, mu, a)


def compute_errors(n_values, T=1.0, mu=1.0, a=0.0):
    """
    Compute L2 and L-infinity errors for various grid resolutions.
    """
    errors_L2 = np.zeros(len(n_values))
    errors_Linf = np.zeros(len(n_values))
    times = np.zeros(len(n_values))
    
    for i, n in enumerate(n_values):
        start_time = time.time()
    
        heat = Heat(mu=mu, a=a, g=g_exact, n=n)
        heat.set_IC(lambda x, y: u_exact(x, y, 0, mu, a))
        
        # Run simulation
        U_snapshots, t = heat.simulate(tf=T, snapshot_stride=1)
        times[i] = time.time() - start_time
        
        # Compute exact solution at final time
        X, Y = heat.get_grid()
        U_exact = u_exact(X, Y, T, mu, a).flatten()
        
        # Compute errors
        error = U_snapshots[-1] - U_exact
        errors_L2[i] = np.sqrt(np.mean(error**2))
        errors_Linf[i] = np.max(np.abs(error))
        
        print(f"n={n}, h={1.0/n:.6f}, L2 Error={errors_L2[i]:.6e}, Linf Error={errors_Linf[i]:.6e}")
    
    return errors_L2, errors_Linf, times


def run_convergence():
    T = 1.0
    mu = 1.0
    a = 1.0
    
    # Test finer mesh sizes
    n_values = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    h_values = 1.0 / n_values 
    errors_L2, errors_Linf, times = compute_errors(n_values, T, mu, a)
    
    # Calculate convergence rates using mesh size ratios
    rates_L2 = np.log(errors_L2[:-1] / errors_L2[1:]) / np.log(h_values[:-1] / h_values[1:])
    rates_Linf = np.log(errors_Linf[:-1] / errors_Linf[1:]) / np.log(h_values[:-1] / h_values[1:])
    
    print("\nConvergence rates:")
    for i, (n, rate_L2, rate_Linf) in enumerate(zip(n_values[:-1], rates_L2, rates_Linf)):
        print(f"  n={n} -> n={n_values[i+1]}: L2 rate={rate_L2:.4f}, Linf rate={rate_Linf:.4f}")
        
    # Plot errors vs mesh size
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].loglog(h_values, errors_L2, 'o-', label='L2 Error')
    ax[0].loglog(h_values, errors_Linf, 'o-', label='Linf Error')
    ax[0].loglog(h_values, errors_L2[0] * (h_values/h_values[0])**2, 'k--', label=r'$\mathcal{O}(h^2)$')
    ax[0].loglog(h_values, errors_L2[0] * (h_values/h_values[0])**4, 'k:', label=r'$\mathcal{O}(h^4)$')
    ax[0].set_xlabel('Step Size $(h)$')
    ax[0].set_ylabel('Error')
    ax[0].set_title('Convergence')
    ax[0].grid(True)
    ax[0].legend()
    
    ax[1].semilogy(n_values, times, 'o-')
    ax[1].set_xlabel('Spatial Resolution (n)')
    ax[1].set_ylabel('Time (s)')
    ax[1].set_title('Computation Time')
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=150)
    return fig


def demonstrate_stability():
    """
    Demonstrate the unconditional stability by using extremely large time steps.
    """
    T = 10
    mu = 1.0
    a = 1.0
    n = 50
    
    r_list = [0.01, 0.1, 1.0, 10.0, 100, 1000]
    
    # Calculate corresponding time steps
    h = 1.0 / n
    k_values = [(r * h**2) / mu for r in r_list]
    
    # Create a figure with TWO subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    
    # First, plot exact solution on the first subplot
    x = y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    U_exact = u_exact(X, Y, T, mu, a)
    
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(X, Y, U_exact, cmap='viridis', edgecolor='none')
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$u(x,y,t)$')
    ax.set_title('Exact Solution')
    
    for i, (r, k) in enumerate(zip(r_list, k_values)):
        print(f"r={r}, k={k:.4f}")
        heat = Heat(mu=mu, a=a, g=g_exact, n=n)
        heat.set_IC(lambda x, y: u0_exact(x, y, mu, a))
        heat.set_k(k)
        
        U_snapshots, t = heat.simulate(tf=T, snapshot_stride=1)
        X, Y = heat.get_grid()
        U = U_snapshots[-1].reshape((n+1, n+1))
        print(f"r={r}, k={k:.4f}, max(U)={np.max(U):.4f}")
        
        
        ax.plot_surface(X, Y, U, cmap=cm.viridis, alpha=0.5)
    
    # Calculate errors for error plot
    errors = []
    for k in k_values:
        heat = Heat(mu=mu, a=a, g=g_exact, n=n)
        heat.set_IC(lambda x, y: u0_exact(x, y, mu, a))
        heat.set_k(k)
        
        U_snapshots, t = heat.simulate(tf=T, snapshot_stride=1)
        X, Y = heat.get_grid()
        U_exact = u_exact(X, Y, T, mu, a).flatten()
        errors.append(np.max(np.abs(U_snapshots[-1] - U_exact)))
    
    error_ax = axes[1]
    error_ax.loglog(r_list, errors, 'o-', label=r'$\max |u - u_{exact}|$')
    error_ax.set_title(r'Error vs Stability Parameter')
    error_ax.set_xlabel(r'$r = \frac{\mu k}{h^2}$')
    error_ax.set_ylabel(r'$\max |u - u_{exact}|$')
    error_ax.grid(True)
    error_ax.legend()
    
    plt.tight_layout()
    plt.savefig('stability_demo.png', dpi=300)
    return fig
    
def animate_solution():
    """
    Create an animation of the solution to visualize its evolution over time.
    """
    T = 1.0
    mu = 1.0
    a = 2.0
    n = 50
    
    heat = Heat(mu=mu, a=a, g=g_exact, n=n)
    heat.set_IC(lambda x, y: u0_exact(x, y, mu, a))
    
    # Simulate with many snapshots
    U_snapshots, t = heat.simulate(tf=T, snapshot_stride=4)
    
    # Animate the solution
    fig, anim = heat.animate_solution(U_snapshots, t, fps=15)
    
    # Save the animation
    anim.save('solution_animation.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
    return fig, anim

def study_parameter_effects():
    """Study the effects of mu and a parameters on the solution"""
    n = 50
    T = 1
    
    # Different parameter combinations to test
    params = [
        {'mu': 0.01, 'a': 10.0, 'label': r'Very low diffusion + high growth ($\mu=0.01, a=10$)'},
        {'mu': 0.1, 'a': 0.0, 'label': r'Low diffusion ($\mu=0.1, a=0$)'},
        {'mu': 1.0, 'a': 0.0, 'label': r'Medium diffusion ($\mu=1.0, a=0$)'},
        {'mu': 5.0, 'a': 0.0, 'label': r'High diffusion ($\mu=5.0, a=0$)'},
        {'mu': 1.0, 'a': 2.0, 'label': r'Growth ($\mu=1.0, a=1$)'},
        {'mu': 1.0, 'a': -2.0, 'label': r'Decay ($\mu=1.0, a=-1$)'},
        {'mu': 10.0, 'a': 0.01, 'label': r'Very high diffusion + low growth ($\mu=10.0, a=0.01$)'}
    ]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(len(params), 2, figsize=(14, 4*len(params)))
    
    for i, param in enumerate(params):
        mu = param['mu']
        a = param['a']
        label = param['label']
        
        # Create heat equation solver
        heat = Heat(mu=mu, a=a, g=g_exact, n=n)
        heat.set_IC(lambda x, y: u0_exact(x, y, mu, a))
        
        # Simulate
        U_snapshots, t = heat.simulate(tf=T, snapshot_stride=10)
        X, Y = heat.get_grid()
        
        # Plot initial state
        U0 = U_snapshots[0].reshape((n+1, n+1))
        c0 = axes[i, 0].contourf(X, Y, U0, 20, cmap='viridis')
        axes[i, 0].set_title(f'{label} - Initial State')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        fig.colorbar(c0, ax=axes[i, 0])
        
        # Plot final state
        Uf = U_snapshots[-1].reshape((n+1, n+1))
        cf = axes[i, 1].contourf(X, Y, Uf, 20, cmap='viridis')
        axes[i, 1].set_title(f'{label} - Final State (t={T})')
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')
        fig.colorbar(cf, ax=axes[i, 1])
    
    plt.tight_layout()
    plt.savefig('parameter_effects.png', dpi=300)
    return fig


def error_vs_timestep():
    """
    Study how error varies with timestep size while keeping spatial resolution fixed
    """
    n = 50 
    mu = 1.0
    a = 1.0  
    T = 1.0
    
    k_values = np.logspace(-5, 0, 25)  
    errors = []
    
    for k in k_values:
        heat = Heat(mu=mu, a=a, g=g_exact, n=n)
        heat.set_IC(lambda x, y: u0_exact(x, y, mu, a))
        heat.set_k(k)
        
        # Ensure we get the solution at exactly T
        stride = max(1, int(T/(100*k)))  # At least 100 snapshots
        U_snapshots, t = heat.simulate(tf=T, snapshot_stride=stride)
        
        X, Y = heat.get_grid()
        U_exact = u_exact(X, Y, T, mu, a).flatten()
        errors.append(np.max(np.abs(U_snapshots[-1] - U_exact)))
        print(f"k={k:.4e}, max error={errors[-1]:.4e}")
    
    errors = np.array(errors)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    # First subplot: Error vs timestep
    ax1.loglog(k_values, errors, 'bo-', label='Numerical Error', linewidth=1.5)
    
    # add k^1 reference line
    ax1.loglog(k_values, errors[0]*(k_values/k_values[0]), 'k:',
                label=r'$\mathcal{O}(k)$ Reference', linewidth=1.5)
    
    # Add k^2 reference line
    k_ref = np.array([min(k_values), max(k_values)])
    ax1.loglog(k_ref, errors[0]*(k_ref/k_values[0])**2, 'k--', 
             label=r'$\mathcal{O}(k^2)$ Reference', linewidth=1.5)
    
    # Add k^3 reference line
    ax1.loglog(k_ref, errors[0]*(k_ref/k_values[0])**3, 'k:',
                label=r'$\mathcal{O}(k^3)$ Reference', linewidth=1.5)
    
    ax1.set_title('Error vs Time Step Size')
    ax1.set_xlabel('Time step size $(k)$')
    ax1.set_ylabel(r'Max Error $\|e\|_\infty$')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()
    
    # Calculate and plot convergence rates
    rates = np.log2(errors[:-1] / errors[1:]) / np.log2(k_values[:-1] / k_values[1:])
    
    ax2.semilogx(k_values[:-1], rates, 'ro-', label='Observed Rate', linewidth=1.5)
    ax2.axhline(y=2, color='k', linestyle='--', label='Expected Rate (2)', linewidth=1.5)
    
    ax2.set_ylim([min(0, min(rates)-0.5), max(4, max(rates)+0.5)])
    ax2.set_title('Observed Convergence Rates')
    ax2.set_xlabel('Time step size $(k)$')
    ax2.set_ylabel('Local Convergence Rate')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    plt.suptitle(f'Temporal Error Analysis $(n={n}, \mu={mu}, a={a})$', y=1.02)
    plt.tight_layout()
    
    # Print statistics
    print("\nConvergence Rate Statistics:")
    print(f"Mean rate: {np.mean(rates):.2f}")
    print(f"Std dev:   {np.std(rates):.2f}")
    print(f"Min rate:  {np.min(rates):.2f}")
    print(f"Max rate:  {np.max(rates):.2f}")
    
    plt.savefig('error_vs_timestep.png', dpi=300, bbox_inches='tight')
    return fig


print("========== RUNNING THEORY VALIDATION ==========")
print("Running convergence study to verify theoretical order...")
run_convergence()

print("Demonstrating unconditional stability...")
demonstrate_stability()

print("Creating animation of solution evolution...")
animate_solution()

print("\n========== CREATING VISUALIZATIONS FOR THEORY SECTION ==========")
print("\n2. Studying parameter effects...")
study_parameter_effects()
print("   Saved to 'parameter_effects.png'")

# Error vs timestep study
print("\n3. Studying error vs timestep...")
error_vs_timestep()
print("   Saved to 'error_vs_timestep.png'")


print("\n========== VISUALIZATIONS COMPLETE ==========")
print("All figures and animations saved to disk.")

plt.show()