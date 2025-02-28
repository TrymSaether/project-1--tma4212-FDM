import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, coo_matrix, eye, kron



# Define a custom colormap for infection and susceptible
colors = ["#00FF00", "#FF0000"]  # Green for susceptible, red for infected
sir_cm = plt.cm.colors.ListedColormap(colors)


class SIR:
    def __init__(self, beta, gamma, muS, muI, n):
        self.n = n
        self.N = (n + 1) ** 2
        self.beta = beta
        self.gamma = gamma
        self.muS = muS
        self.muI = muI
        self.dn = 1.0 / n

        self.L = self.laplacian(n, self.dn)
        self.S = np.ones(self.N)
        self.I = np.zeros(self.N)

        self.x = self.y = np.arange(n + 1) * self.dn
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.X, self.Y = self.X.flatten(), self.Y.flatten()

    @staticmethod
    def laplacian(n, dn):
        diag = -2 * np.ones(n + 1)
        diag[0] = diag[-1] = -1
        off = np.ones(n)
        L = diags([off, diag, off], [-1, 0, 1], shape=(n + 1, n + 1)) / dn**2
        I = eye(n + 1)
        return kron(I, L) + kron(L, I)

    def reaction(self, S, I):
        inf_term = self.beta * S * I
        dS_react = -inf_term
        dI_react = inf_term - self.gamma * I
        return dS_react, dI_react

    def diffusion(self, L, S, I):
        dS_diff = L * S
        dI_diff = L * I
        return dS_diff, dI_diff

    def set_random_infection(self, rate, prob=0.1):
        mask = np.random.rand(self.N) < prob
        self.I[mask] = rate
        self.S[mask] = 1.0 - rate
        # self.plot_initial_conditions()

    def set_infection(self, x0, y0, r, rate):
        mask = ((self.X - x0) ** 2 + (self.Y - y0) ** 2) <= r**2
        self.I[mask] += rate
        self.S[mask] += 1.0 - rate

    def simulate(self, dt, tf, snapshot_stride=1, event=False, **kwargs):
        T = int(tf / dt)
        Sd, Id, t = [], [], []
        for n in range(T + 1):
            if event == True:
                t_ev, x0, y0, r, rate = kwargs.values()
                if n * dt >= t_ev:
                    self.set_infection(x0, y0, r, rate)
                    event = False

            if n % snapshot_stride == 0:
                Sd.append(self.S.copy())
                Id.append(self.I.copy())
                t.append(n * dt)

            dS_react, dI_react = self.reaction(self.S, self.I)
            dS_diff, dI_diff = self.diffusion(self.L, self.S, self.I)

            self.S += dt * (dS_react + self.muS * dS_diff)
            self.I += dt * (dI_react + self.muI * dI_diff)

        X, Y = self.get_grid()
        return (Sd, Id), (X, Y), t

    def get_grid(self):
        X = self.X.reshape((self.n + 1, self.n + 1))
        Y = self.Y.reshape((self.n + 1, self.n + 1))
        return X, Y

    def get_solution(self):
        return self.Sd, self.Id, self.t

    def get_L(self):
        return self.L

    def plot_initial_conditions(self):
        X, Y = self.get_grid()
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111)
        inf = ax.scatter(X, Y, c=self.I, cmap=sir_cm)
        # Add a legend and infection rate information
        cbar = plt.colorbar(inf, ticks=[0, 1])
        cbar.set_ticklabels(["Susceptible", "Infected"])
        infection_text = f"Infection rate: {self.I.sum() / self.N:.2%}"
        ax.text(
            0.05,
            0.05,
            infection_text,
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=1),
        )
        ax.set_title("Initial Infection")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        plt.show()

    def plot_S(self, tn):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            self.X, self.Y, self.Sd[tn].reshape((self.n + 1, self.n + 1)), cmap="plasma"
        )
        ax.set_title(f"Susceptible, t={self.t[tn]:.3f}")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$S \\%$")
        ax.set_zlim(0, max(0.01, self.Sd[tn].max() * 1.1))
        plt.show()

    def plot_L(self):
        plt.imshow(self.L.toarray() * self.dn**2, cmap="plasma")
        plt.colorbar()
        plt.title("Laplacian Matrix")
        plt.show()


N = 100
tf = 15
dt = 0.001
snapshot_stride = 10

beta = 3.0
gamma = 1.0

muS = 0.001
muI = 0.001
rate = 0.25

# prob = 0.05


model = SIR(beta, gamma, muS, muI, N)

# model.set_random_infection(rate, prob)
model.set_infection(x0=0.1, y0=0.1, r=0.02, rate=rate)
model.set_infection(x0=0.8, y0=0.8, r=0.02, rate=rate)

event = {"t_ev": 8.1, "x0": 0.85, "y0": 0.85, "r": 0.1, "rate": rate}


(S, I), (X, Y), t = model.simulate(
    dt=dt, tf=tf, snapshot_stride=snapshot_stride, event=True, **event
)
I_2d = I[0].reshape((N + 1, N + 1))

fig = plt.figure(figsize=(10, 8), dpi=150)

ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, I_2d, cmap="plasma", edgecolor="none")

ax.set_title(f"Infected, t={t[0]:.3f}")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$I \\%$")

current_surf = [surf]


def init():
    return current_surf


def update(frame):
    global current_surf
    for c in current_surf:
        c.remove()
    current_surf = []

    I_2d = I[frame].reshape((N + 1, N + 1))

    scale_cmap = 1.0 / I_2d.max()
    new_surf = ax.plot_surface(
        X,
        Y,
        I_2d,
        cmap="plasma",
        edgecolor="none",
        facecolors=plt.cm.plasma(I_2d * scale_cmap),
    )

    current_surf.append(new_surf)

    ax.set_title(f"Infected, t={t[frame]:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Infected fraction")
    ax.set_zlim(0, 1)

    return current_surf


anim = FuncAnimation(
    fig, update, frames=len(I), init_func=init, blit=False, interval=1, repeat=True
)


# Create visualization of the SIR model progression with S and I side by side
fig = plt.figure(figsize=(16, 10))

# Select specific time points to show the progression
time_indices = [int(len(t)/10), int(len(t)/5), int(len(t)*2/5), int(len(t)*3/5), int(len(t)*4/5), -1]
time_points = [t[i] for i in time_indices]

for i, idx in enumerate(time_indices):
    # Plot for Susceptible (S)
    ax1 = fig.add_subplot(2, 3, i+1, projection='3d')
    
    # Get the 2D representation of susceptible data
    S_2d = S[idx].reshape((N+1, N+1))
    
    # Plot S with viridis colormap
    surf_s = ax1.plot_surface(
        X, Y, S_2d, 
        cmap='viridis',
        edgecolor='none',
        alpha=1,
        antialiased=True
    )
    
    # Set plot properties
    ax1.set_title(f"t = {t[idx]:.2f}", fontsize=12)
    ax1.set_xlabel("x", fontsize=10)
    ax1.set_ylabel("y", fontsize=10)
    ax1.set_zlabel("S %", fontsize=10)
    ax1.set_zlim(0, 1.0)
    
    # Add annotation with susceptible statistics
    s_percent = S[idx].sum() / len(S[idx]) * 100
    ax1.text2D(0.05, 0.95, f"Susceptible: {s_percent:.2f}%", 
               transform=ax1.transAxes, fontsize=10,
               bbox=dict(facecolor='white', alpha=1))

# Create a separate figure for Infected plots
fig2 = plt.figure(figsize=(16, 10))

for i, idx in enumerate(time_indices):
    # Plot for Infected (I)
    ax2 = fig2.add_subplot(2, 3, i+1, projection='3d')
    
    # Get the 2D representation of infection data
    I_2d = I[idx].reshape((N+1, N+1))
    
    # Plot I with plasma colormap
    surf_i = ax2.plot_surface(
        X, Y, I_2d, 
        cmap='plasma',
        edgecolor='none',
        antialiased=True
    )
    
    # Set plot properties
    ax2.set_title(f"t = {t[idx]:.2f}", fontsize=12)
    ax2.set_xlabel("x", fontsize=10)
    ax2.set_ylabel("y", fontsize=10)
    ax2.set_zlabel("I %", fontsize=10)
    ax2.set_zlim(0, 1.0)
    
    # Add annotation with infection statistics
    i_percent = I[idx].sum() / len(I[idx]) * 100
    ax2.text2D(0.05, 0.95, f"Infected: {i_percent:.2f}%", 
               transform=ax2.transAxes, fontsize=10,
               bbox=dict(facecolor='white'))

# Add titles and adjust layout
fig.suptitle("Progression of Susceptible Population Over Time", fontsize=16)
fig2.suptitle("Progression of Infected Population Over Time", fontsize=16)

fig.tight_layout()
fig2.tight_layout()

# fig.savefig("susceptible_progression.png", dpi=300)
# fig2.savefig("infected_progression.png", dpi=300)
plt.show()
