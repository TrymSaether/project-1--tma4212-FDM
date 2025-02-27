import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, coo_matrix, eye, kron

class SIR:
    def __init__(self, beta, gamma, muS, muI, N):
        self.N = N        
        self.beta = beta
        self.gamma = gamma
        self.muS = muS
        self.muI = muI
        self.dn = 1.0 / N
        
        self.L = self.laplacian(N, self.dn)
        self.S = np.ones((N + 1, N + 1))
        self.I = np.zeros((N + 1, N + 1))
        
        self.x = self.y = np.arange(N + 1) * self.dn
        self.X, self.Y = np.meshgrid(self.x, self.y)
    
    def laplacian(N, dn):
        diag = -2 * np.ones(N + 1)
        diag[0] = diag[-1] = -1
        off = np.ones(N)
        L = diags([off, diag, off], [-1, 0, 1], shape=(N + 1, N + 1))/ dn**2
        I = eye(N + 1)
        return kron(I, L) + kron(L, I)
    
    def reaction(self, S, I):
        inf_term = self.beta * S * I
        dS_react = -inf_term
        dI_react = inf_term - self.gamma * I
        return dS_react, dI_react

    def diffusion(self, L, S, I):
        dS_diff = L.dot(S)
        dI_diff = L.dot(I)
        return dS_diff, dI_diff
    
    def set_infection(self, x0, y0, r, rate):       
        mask = ((self.X - x0)**2 + (self.Y - y0)**2) < r**2
        self.I[mask] = rate
        self.S[mask] = 1.0 - rate
    
    def simulate(self, dt, tf, snapshot_stride=1):
        T = int(tf / dt) 
        S_list, I_list, times = [], [], []
        for n in range(T + 1):
            if n % snapshot_stride == 0:
                S_list.append(self.S.copy())
                I_list.append(self.I.copy())
                times.append(n * dt)
            
            dS_react, dI_react = self.reaction(self.S, self.I)
            dS_diff, dI_diff = self.diffusion(self.L, self.S.flatten(), self.I.flatten())

            self.S += dt * (dS_react + self.muS * dS_diff).reshape((self.N + 1, self.N + 1))
            self.I += dt * (dI_react + self.muI * dI_diff).reshape((self.N + 1, self.N + 1))

        return S_list, I_list, times

        
    

    
def laplacian(N, dn):
    diag = -2 * np.ones(N + 1)
    diag[0] = diag[-1] = -1
    off = np.ones(N)
    L = diags([off, diag, off], [-1, 0, 1], shape=(N + 1, N + 1))/ dn**2
    I = eye(N + 1)
    return kron(I, L) + kron(L, I)

M = 5
dm = 1.0 / M

plt.imshow(laplacian(M, dm).todense(), cmap="plasma")
plt.colorbar()
plt.show()

def _build_laplacian(Mx, My, dx, dy):
    """Construct the 2D Laplacian matrix with Neumann boundary conditions."""
    Nx = (Mx + 1) * (My + 1)

    def idx(i, j):
        return i * (My + 1) + j

    cx = 1.0 / dx**2
    cy = 1.0 / dy**2

    data, row, col = [], [], []

    for i in range(Mx + 1):
        for j in range(My + 1):
            center = idx(i, j)

            # Start with diagonal term
            diag = 0

            # Handle x-direction connections
            if i > 0:  # Has left neighbor
                row.append(center)
                col.append(idx(i - 1, j))
                data.append(cx)
                diag -= cx

            if i < Mx:  # Has right neighbor
                row.append(center)
                col.append(idx(i + 1, j))
                data.append(cx)
                diag -= cx

            # Handle y-direction connections
            if j > 0:  # Has bottom neighbor
                row.append(center)
                col.append(idx(i, j - 1))
                data.append(cy)
                diag -= cy

            if j < My:  # Has top neighbor
                row.append(center)
                col.append(idx(i, j + 1))
                data.append(cy)
                diag -= cy

            # Add diagonal term
            row.append(center)
            col.append(center)
            data.append(diag)

    return coo_matrix((data, (row, col)), shape=(Nx, Nx)).tocsr()


M = 5
N = 5
dx = 1.0 / M
dn = dy = 1.0 / N

L1 = laplacian(N,dn)
L2 = _build_laplacian(M, N, dx, dy)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
im0 = ax[0].imshow(L1.todense(), cmap="plasma")
ax[0].set_title("Laplacian using kronecker")
plt.colorbar(im0, ax=ax[0])

im1 = ax[1].imshow(L2.todense(), cmap="plasma")
ax[1].set_title("Laplacian using sparse matrix")
plt.colorbar(im1, ax=ax[1])

plt.show()


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
    dx = (x1 - x0) / Mx
    dy = (y1 - y0) / My
    Nx = (Mx + 1) * (My + 1)
    Nsteps = int(T / dt)

    L = laplacian(Mx, dx)

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
            if (xcoord - 0.20) ** 2 + (ycoord - 0.20) ** 2 < 0.005:
                I[k] = 0.01
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

    plt.imshow(I.reshape((Mx + 1, My + 1)), cmap="plasma")
    plt.colorbar()
    plt.title("Initial infected fraction")
    plt.show()

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

        # # Update
        # S += dt * (dS_react + muS * dS_diff)
        # I += dt * (dI_react + muI * dI_diff)

        S += dt * (-beta * S * I + muS * L * S)
        I += dt * (beta * S * I - gamma * I + muI * L * I)

    return S_list, I_list, times, (Mx, My)


Mx = 50
My = 50
dt = 0.001
T = 30
beta = 3.0
gamma = 0.50
muS = 0.001
muI = 0.001


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


anim = FuncAnimation(fig, update, frames=len(I_list), init_func=init, blit=False, interval=1, repeat=True)
plt.show()
# anim.save("3D_infected_animation.mp4", fps=30, extra_args=["-vcodec", "libx264"])
