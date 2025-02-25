import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, eye, kron

class SIRSimulation:
    def __init__(self, L=10.0, Nx=50, Ny=50, T=10, dt=0.005, initial_infection=0.1,
                 beta=3, gamma=1, mu_s=0.1, mu_i=0.1):
        self.L = L
        self.Nx, self.Ny = Nx, Ny
        self.h = L / Nx
        self.T = T
        self.dt = dt
        self.Nt = int(T / dt)
        self.beta = beta
        self.gamma = gamma
        self.mu_s = mu_s
        self.mu_i = mu_i

        self.x = np.linspace(0, L, Nx)
        self.y = np.linspace(0, L, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.S = np.ones(Nx * Ny)
        self.I = np.zeros(Nx * Ny)
        self.R = np.zeros(Nx * Ny)

        for i in range(Nx):
            for j in range(Ny):
                idx_flat = i * Ny + j
                if (self.x[i] - L/2)**2 + (self.y[j] - L/2)**2 < 0.2:
                    self.I[idx_flat] = initial_infection
                    self.S[idx_flat] -= initial_infection

        self.L_matrix = self.laplacian(Nx, Ny)

    
    def laplacian(self, n, m):
        Mi = n       # Number of inner points in each direction
        Mi2 = n * m   # Number of inner points in total

        # Construct a sparse A-matrix
        B = diags([1,-4,1],[-1,0,1],shape=(Mi, Mi), format="csr")
        A = kron(eye(Mi), B)
        C = diags([1,1],[-Mi,Mi],shape=(Mi2, Mi2), format="csr")
        A = (A+C).tocsr() # Konverter til csr-format (necessary for spsolve) 
        return A/self.h**2

    def step(self):
        infection = self.beta * self.S * self.I
        recovery = self.gamma * self.I

        diffusion_S = self.mu_s * self.L_matrix.dot(self.S)
        diffusion_I = self.mu_i * self.L_matrix.dot(self.I)

        dS = -infection + diffusion_S
        dI = infection - recovery + diffusion_I
        dR = recovery

        self.S += self.dt * dS
        self.I += self.dt * dI
        self.R += self.dt * dR

        self.S = np.maximum(self.S, 0)
        self.I = np.maximum(self.I, 0)
        self.R = np.maximum(self.R, 0)

    def animate(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            for _ in range(10):
                self.step()
            ax.clear()
            ax.plot_surface(self.X, self.Y, self.I.reshape(self.Nx, self.Ny), cmap='hot', edgecolor='none')
            ax.set_zlim(0, 1)
            ax.set_title(f'Infected Population at t={frame*self.dt*10:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('Infected Fraction')

        anim = FuncAnimation(fig, update, frames=self.Nt//10, interval=20)
        plt.show()
    

if __name__ == "__main__":
    sim = SIRSimulation()
    sim.animate()