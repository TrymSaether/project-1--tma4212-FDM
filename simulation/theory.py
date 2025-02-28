import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve, factorized

import scienceplots

plt.style.use("science")


class Heat:
    def __init__(self, mu, a, g, n=25):
        self.n = n
        self.N = (n + 1) ** 2
        self.h = 1.0 / n
        self.k = 1.0 / n
        
        self.mu, self.a = mu, a
        self.r = mu * self.k / (self.h**2)

        self.g = g

        self.x = self.y = np.arange(n + 1) * self.h
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Xf, self.Yf = self.X.flatten(), self.Y.flatten()

        self.U = np.zeros(self.N)
        self.L = self.laplacian(n)


    @staticmethod
    def laplacian(n):
        diag = -2 * np.ones(n + 1)
        diag[0] = diag[-1] = -1
        off = np.ones(n)
        Lh = diags([off, diag, off], [-1, 0, 1], shape=(n + 1, n + 1))
        Ih = eye(n + 1)
        return kron(Ih, Lh) + kron(Lh, Ih)
    
    def set_BC(self, U, t=0):
        if np.shape(U) == (self.N,):
            U = U.reshape((self.n + 1, self.n + 1))

        g = self.g
        X, Y = self.get_grid()

        U[0, :] = g(X[0, :], Y[0, :], t)  # Bottom
        U[-1, :] = g(X[-1, :], Y[-1, :], t)  # Top
        U[:, 0] = g(X[:, 0], Y[:, 0], t)  # Left
        U[:, -1] = g(X[:, -1], Y[:, -1], t)  # Right
        return U.flatten()

    def set_IC(self, u0):
        self.U = u0(self.Xf, self.Yf).flatten()

    def simulate(self, tf=1.0, snapshot_stride=1):
        T = int(tf / self.k)
        if T < 1:
            T = int(self.k / tf)
        I = eye(self.N)
        
        A = factorized((I - (self.r / 2) * self.L).tocsc())
        B = I + (self.r / 2) * self.L + (self.k * self.a) * I
    
        self.U = self.set_BC(self.U, 0)
        
        Ud, t = [], []
        for n in range(1, T + 1):
            if n % snapshot_stride == 0:
                Ud.append(self.U.copy())
                t.append(n * self.k)
    
            # Predictor
            b = B @ self.U
            U_ = A(b)
            U_ = self.set_BC(U_, n * self.k)

            # Corrector
            self.U = U_ + (self.k * self.a / 2) * (U_ - self.U)
        return np.array(Ud), np.array(t)

    def get_grid(self):
        return self.X, self.Y

    def get_solution(self):
        return self.U.reshape((self.n + 1, self.n + 1))

    def get_L(self):
        return self.L

    def get_h(self):
        return self.h

    def get_r(self):
        return self.r

    def get_k(self):
        return self.k

    def set_k(self, k):
        self.k = k
        
    def set_h(self, h):
        self.h = h
        self.n = int(1.0 / h)
        self.N = (self.n + 1) ** 2
        self.r = self.mu * self.k / (self.h**2)
        self.x = self.y = np.arange(self.n + 1) * self.h
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Xf, self.Yf = np.meshgrid(self.x, self.y)
        self.L = self.laplacian(self.n)
        
    def plot_solution(self, t_idx=0, U_snapshots=None, t=None):
        if U_snapshots is None:
            U_snap = self.U
            t_lbl = "current state"
        else:
            U_snap = U_snapshots[t_idx]
            t_lbl = f"t = {t[t_idx]:.4f}"

        X, Y = self.get_grid()
        U = U_snap.reshape((self.n + 1, self.n + 1))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, U, cmap="viridis", edgecolor="none")

        ax.set_title(f"Heat Equation Solution at {t_lbl}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u(x,y,t)")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.tight_layout()
        plt.show()

    def animate_solution(self, U_snapshots, t, fps=10):
        X, Y = self.get_grid()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        U_shaped = U_snapshots[0].reshape((self.n + 1, self.n + 1))
        surf = ax.plot_surface(X, Y, U_shaped, cmap="viridis", edgecolor="none")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u(x,y,t)")

        # Set common z-limits for all frames
        z_min = np.min(U_snapshots)
        z_max = np.max(U_snapshots)
        margin = 0.1 * (z_max - z_min)
        ax.set_zlim(z_min - margin, z_max + margin)

        title = ax.set_title(f"t = {t[0]:.4f}")

        current_surf = [surf]

        def init():
            return current_surf

        def update(frame):
            for s in current_surf:
                s.remove()
            current_surf.clear()

            # Plot new surface
            U_shaped = U_snapshots[frame].reshape((self.n + 1, self.n + 1))
            new_surf = ax.plot_surface(X, Y, U_shaped, cmap="viridis", edgecolor="none")
            current_surf.append(new_surf)
            title.set_text(f"t = {t[frame]:.4f}")
            return current_surf

        anim = FuncAnimation(
            fig,
            update,
            frames=len(t),
            init_func=init,
            blit=False,
            interval=1000 / fps,
            repeat=True,
        )
        plt.tight_layout()
        return fig, anim


def g0(x, y, t): 
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2 * np.pi**2 * t)

def u0(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def heat_example_1(Heat, g0, u0):
    mu = 1.0
    a = 1.0

    heat = Heat(mu=1.0, a=1.0, n=25, g=g0)
    heat.set_initial_condition(u0)
    U_snapshots, t = heat.simulate(tf=10.0, snapshot_stride=10)
    heat.plot_solution(t_idx=0, U_snapshots=U_snapshots, t=t)
    heat.animate_solution(U_snapshots, t)

def heat_error_analysis_1(Heat, g0, u0):
    n = 25
    a_l = np.linspace(-1, 1, 5)
    mu_l = np.linspace(0.1, 3, 5)
    k_l = [0.001, 0.001, 0.01, 0.1, 1.0]

    errors = np.zeros((len(a_l), len(mu_l), len(k_l)))
    for i, a in enumerate(a_l):
        for j, mu in enumerate(mu_l):
            for k, k_ in enumerate(k_l):
                heat = Heat(mu=mu, a=a, g=g0, n=n)
                heat.set_initial_condition(u0)
                heat.set_k(k_)
                U_snapshots, t = heat.simulate(tf=1.0, snapshot_stride=1)
                X, Y = heat.get_grid()
                U_exact = g0(X, Y, 1.0)
                errors[i, j, k] = np.linalg.norm(U_snapshots[-1] - U_exact.flatten())

    n_a, n_mu = len(a_l), len(mu_l)

    fig, ax = plt.subplots(n_a, n_mu, figsize=(n_mu * 5, n_a * 5))
    for i in range(n_a):
        for j in range(n_mu):
            ax[i, j].plot(k_l, errors[i, j], marker="o")
            ax[i, j].set_title(f"a={a_l[i]}, mu={mu_l[j]}")
            ax[i, j].set_xscale("log")
            ax[i, j].set_yscale("log")
            ax[i, j].set_xlabel("k")
            ax[i, j].set_ylabel("Error")
        
    plt.tight_layout()
    plt.show()
    
    


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

fig, (ax_k, ax_h) = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

n = 50
h = 1.0/n
mu = 1.0
a = 1.0 
T = 1.0

k_values = np.logspace(-3, 2, 30)
E_k = []
for k in k_values:
    heat = Heat(mu=1.0, a=1.0, n=50, g=g0)
    heat.set_IC(u0)
    heat.set_k(k)
    h = heat.get_h()
    
    stride = max(1, int(T/(100*k))) 
    U_snapshots, t = heat.simulate(tf=T, snapshot_stride=stride)
    
    X, Y = heat.get_grid()
    U_exact_k = u_exact(X, Y, T, mu, a).flatten()
    E_k.append(np.max(np.abs(U_snapshots[-1] - U_exact_k)))
    print(f"k={k:.4e}, max error={E_k[-1]:.4e}")
    

ax_k.loglog(k_values, np.array(E_k), 'bo-', label='$\\mathcal{O}(k^p)$', linewidth=1.5)

ax_k.loglog(k_values, k_values, 'k--', label=r'$\mathcal{O}(k)$', alpha=0.5)
ax_k.loglog(k_values, k_values**2, 'k:', label=r'$\mathcal{O}(k^2)$')
ax_k.loglog(k_values, k_values**3, 'k-.', label=r'$\mathcal{O}(k^3)$', alpha=0.5)

ax_k.set_xlabel('Time Step $(k)$', fontsize=14)
ax_k.set_ylabel('Error', fontsize=14)
ax_k.set_title(f'Error vs Time Step', fontsize=20)
ax_k.legend(fontsize=12)
ax_k.grid(True)


# =========== H-convergence ===========
n_values = np.logspace(1, 2.5, 30, dtype=int)
h_values = 1.0/n_values
E_h = []

for n in n_values:
    k = 1.0/n
    heat = Heat(mu=1.0, a=1.0, n=n, g=g0)
    heat.set_IC(u0)
    
    stride = max(1, int(T/(100*k)))
    U_snapshots, t = heat.simulate(tf=T, snapshot_stride=stride)
    
    X, Y = heat.get_grid()
    U_exact = u_exact(X, Y, T, mu, a).flatten()
    E_h.append(np.max(np.abs(U_snapshots[-1] - U_exact)))
    print(f"n={n}, h={1.0/n:.4e}, k={k:.4e}, max error={E_h[-1]:.4e}")

E_h = np.array(E_h)

ax_h.loglog(h_values, E_h, 'ro-', label='$\\mathcal{O}(h^p)$', linewidth=1.5)
ax_h.loglog(h_values, h_values, 'k--', label=r'$\mathcal{O}(h)$', alpha=0.5)
ax_h.loglog(h_values, h_values**2, 'k:', label=r'$\mathcal{O}(h^2)$')

ax_h.set_xlabel('Step Size $(h)$', fontsize=14)
ax_h.set_ylabel('Error', fontsize=14)
ax_h.set_title('Error vs Grid Size', fontsize=20)
ax_h.legend(fontsize=10)
ax_h.grid(True)

plt.tight_layout()
plt.savefig('heat_convergence.png', dpi=300)
plt.show()

