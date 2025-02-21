import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import coo_matrix, csr_matrix


def build_laplacian_2d_neumann_like(Mx, My, dx, dy):
    """Build 2D Laplacian matrix with frozen boundaries"""
    Nx = (Mx + 1) * (My + 1)
    idx = lambda i, j: i * (My + 1) + j
    cx, cy = 1 / dx**2, 1 / dy**2

    rows, cols, vals = [], [], []
    for i in range(Mx + 1):
        for j in range(My + 1):
            r = idx(i, j)
            if i == 0 or i == Mx or j == 0 or j == My:
                rows.append(r)
                cols.append(r)
                vals.append(0.0)
            else:
                # Center point
                rows.append(r)
                cols.append(r)
                vals.append(-2 * (cx + cy))
                # Adjacent points
                for ii, jj, v in [
                    (i - 1, j, cx),
                    (i + 1, j, cx),
                    (i, j - 1, cy),
                    (i, j + 1, cy),
                ]:
                    rows.append(r)
                    cols.append(idx(ii, jj))
                    vals.append(v)

    return coo_matrix((vals, (rows, cols)), shape=(Nx, Nx)).tocsr()


def simulate_sir_diffusion_2d_fast(
    Mx=50,
    My=50,
    x0=0.0,
    x1=1.0,
    y0=0.0,
    y1=1.0,
    beta=3.0,
    gamma=1.0,
    muS=0.001,
    muI=0.001,
    dt=0.0002,
    T=0.5,
    snapshot_stride=50,
):
    """
    SIR-diffusion simulation using:
      - Flattened 2D arrays (1D vectors)
      - Sparse Laplacian matrix for diffusion
      - Vectorized updates
    """
    dx = (x1 - x0) / Mx
    dy = (y1 - y0) / My
    Nx = (Mx + 1) * (My + 1)
    Nsteps = int(T / dt)

    L = build_laplacian_2d_neumann_like(Mx, My, dx, dy)

    def idx(i, j):
        return i * (My + 1) + j

    # Initialize S, I
    S = np.zeros(Nx, dtype=float)
    I = np.zeros(Nx, dtype=float)
    
    # Initial conditions
    for i in range(Mx + 1):
        for j in range(My + 1):
            k = idx(i, j)
            xcoord = x0 + i * dx
            ycoord = y0 + j * dy

            # small circle of infection near center
            if (xcoord - 0.5) ** 2 + (ycoord - 0.5) ** 2 < 0.01:
                I[k] = 0.1
                S[k] = 1.0 - I[k]
            else:
                I[k] = 0.0
                S[k] = 1.0

            # Small but infected region to the top right
            if (xcoord - 0.8) ** 2 + (ycoord - 0.8) ** 2 < 0.005:
                I[k] = 0.1
                S[k] = 1.0 - I[k]

            # Very Small but Highly infected region to the bottom left
            if (xcoord - 0.1) ** 2 + (ycoord - 0.1) ** 2 < 0.005:
                I[k] = 0.5
                S[k] = 1.0 - I[k]

            # Neumann boundary conditions
            if i == 0 or i == Mx or j == 0 or j == My:
                S[k] = 0.0
                I[k] = 0.0

    
    S_list, I_list, times = [], [], []
    
    
    for n in range(Nsteps + 1):
        if n % snapshot_stride == 0:
            S_list.append(S.copy())
            I_list.append(I.copy())
            times.append(n * dt)

        # Reaction
        inf_term = beta * S * I
        dS_react = -inf_term
        dI_react = inf_term - gamma * I

        # Diffusion
        dS_diff = L.dot(S)
        dI_diff = L.dot(I)

        # Update
        S += dt * (dS_react + muS * dS_diff)
        I += dt * (dI_react + muI * dI_diff)

    return S_list, I_list, times, (Mx, My)


Mx = 50
My = 50
dt = 0.0002
T = 30
beta = 3.0
gamma = 0.5
muS = 0.01
muI = 0.01

# 1) Run the simulation
S_list, I_list, times, (Mx, My) = simulate_sir_diffusion_2d_fast(
    Mx=Mx,
    My=My,
    beta=beta,
    gamma=gamma,
    muS=muS,
    muI=muI,
    dt=dt,
    T=T,
    snapshot_stride=10,
)

x_vals = np.linspace(0, 1, Mx + 1)
y_vals = np.linspace(0, 1, My + 1)
X, Y = np.meshgrid(x_vals, y_vals)
X = X.T
Y = Y.T

fig = plt.figure(figsize=(6, 5), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$I$")


# Initial data
I_2d = I_list[0].reshape((Mx + 1, My + 1))
surf = ax.plot_surface(X, Y, I_2d, cmap="plasma", edgecolor="none")
ax.set_title(f"Infected, t={times[0]:.3f}")
cb = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1, label="Infected fraction")


current_surf = [surf]


def init():
    return current_surf


def update(frame):
    global current_surf
    # Remove old surface
    for c in current_surf:
        c.remove()
    current_surf = []

    I_2d = I_list[frame].reshape((Mx + 1, My + 1))

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