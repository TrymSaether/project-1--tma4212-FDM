import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, coo_matrix, eye, kron

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
        dS_diff = L*S
        dI_diff = L*I
        return dS_diff, dI_diff
    
    def set_infection(self, x0, y0, r, rate):       
        mask = ((self.X - x0)**2 + (self.Y - y0)**2) < r**2
        self.I[mask] = rate
        self.S[mask] = 1.0 - rate
    
    def simulate(self, dt, tf, snapshot_stride=1):
        T = int(tf / dt) 
        Sd, Id, t = [], [], []
        for n in range(T + 1):
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
        X = self.X.reshape((self.n+1, self.n+1))
        Y = self.Y.reshape((self.n+1, self.n+1))
        return X, Y
        
    def get_solution(self):
        return self.Sd, self.Id, self.t

N = 50
tf = 30
dt = 0.001

beta = 3.0
gamma = 1

muS = 0.001
muI = 0.001

model = SIR(beta, gamma, muS, muI, N)
model.set_infection(x0=0.5, y0=0.5, r=0.1, rate=0.01)

(S, I), (X, Y), t = model.simulate(dt=0.001, tf=30, snapshot_stride=10)
I_2d = I[0].reshape((N + 1, N + 1))

fig = plt.figure(figsize=(10, 8), dpi=150)

ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, I_2d, cmap="plasma", edgecolor="none")

ax.set_title(f"Infected, t={t[0]:.3f}")
ax.set_xlabel("$x$"); ax.set_ylabel("$y$"); ax.set_zlabel("$I \\%$")

current_surf = [surf]

def init():
    return current_surf

def update(frame):
    global current_surf
    # Remove old surface
    for c in current_surf:
        c.remove()
    current_surf = []

    I_2d = I[frame].reshape((N + 1, N + 1))

    new_surf = ax.plot_surface(X, Y, I_2d, cmap="plasma", edgecolor="none")
    current_surf.append(new_surf)

    ax.set_title(f"Infected, t={t[frame]:.3f}")
    ax.set_zlim(0, max(0.01, I_2d.max() * 1.1))  # adjust as needed
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Infected fraction")

    return current_surf


anim = FuncAnimation(fig, update, frames=len(I), init_func=init, blit=False, interval=1, repeat=True)
plt.show()

# anim.save("3D_infected_animation.mp4", fps=30, extra_args=["-vcodec", "libx264"])
