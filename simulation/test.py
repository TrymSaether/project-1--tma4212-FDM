import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.sparse import kron
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class SIR:
    def __init__(self, M=50, dt=0.001, T=0.1, beta=3.0, gamma=1.0, mu_s=0.1, mu_i=0.5):
        self.M = M
        self.N = (M + 1) ** 2

        self.dt = dt
        self.nsteps = int(T / dt)
        self.beta = beta
        self.gamma = gamma
        self.mu_s = mu_s
        self.mu_i = mu_i
        self.h = 1.0 / (M+1)
        self.L = self._build_laplacian(M+1)

        self.S = np.ones(self.N)
        self.I = np.zeros(self.N)

    def set_initial_conditions(self, I0):
        self.I = I0
        self.S = 1 - I0

    def _build_laplacian(self, m):
        n = m - 1
        a = diags([-1, 4, -1], [-1, 0, 1], shape=(n, n))
        b = kron(eye(m-1), a)
        c = diags([1,1],[-n, n],shape=(n**2, n**2))
        return b + c

    def _step(self):
        S = self.S
        I = self.I
        
        dS = mu_s * self.L.dot(S) / self.h**2
        dI = mu_i * self.L.dot(I) / self.h**2

        self.S += self.dt * (S - self.beta * S * I + dS)
        self.I += self.dt * (I + self.beta * S * I - self.gamma * I + dI)

    def solve(self):
        for _ in range(self.nsteps):
            self._step()
        return self.S, self.I

    def simulate(self):
        S_list, I_list, times = [], [], []
        for n in range(self.nsteps + 1):
            if n % 10 == 0:
                S_list.append(self.S.copy())
                I_list.append(self.I.copy())
                times.append(n * self.dt)
            self._step()
        return S_list, I_list, times


M = 50
dt = 0.001
T = 5
beta = 3.0
gamma = 1.0
mu_s = 0.0001
mu_i = 0.0001

sir = SIR(M=M, dt=dt, T=T, beta=beta, gamma=gamma, mu_s=mu_s, mu_i=mu_i)
I0 = np.zeros((M, M), dtype=float)


I0[0:10, 0:10] = 1

sir.set_initial_conditions(I0.ravel())
S_list, I_list, times = sir.simulate()

x_vals = np.linspace(0, 1, M)
y_vals = np.linspace(0, 1, M)
X, Y = np.meshgrid(x_vals, y_vals)

fig = plt.figure(figsize=(6, 5), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$I$")

# Initial data
I_2d = I_list[0].reshape((M, M))
surf = ax.plot_surface(X, Y, I_2d, cmap="plasma", edgecolor="none")
ax.set_title(f"Infected, t={times[0]:.3f}")

current_surf = [surf]


def init():
    return current_surf


def update(frame):
    global current_surf
    # Remove old surface
    for c in current_surf:
        c.remove()
    current_surf = []

    I_2d = I_list[frame].reshape((M, M))

    new_surf = ax.plot_surface(X, Y, I_2d, cmap="plasma", edgecolor="none")
    current_surf.append(new_surf)

    ax.set_title(f"Infected, t={times[frame]:.3f}")
    ax.set_zlim(0, max(0.01, I_2d.max() * 1.1))  # adjust as needed
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Infected fraction")

    return current_surf


anim = FuncAnimation(
    fig, update, frames=len(I_list), init_func=init, blit=False, interval=1, repeat=True
)
plt.show()
# anim.save("3D_infected_animation.mp4", fps=30, extra_args=["-vcodec", "libx264"])
