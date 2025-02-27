import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, eye, kron

class SIRSimulation:
    def __init__(self, L=1.0, M=6, T=10, dt=0.001, initial_infection=0.01,
                 beta=3, gamma=1, mu_s=0.001, mu_i=0.001):
        self.L = L
        self.M = M
        self.h = L / M
        self.T = T
        self.dt = dt
        self.Nt = int(T / dt)
        self.beta = beta
        self.gamma = gamma
        self.mu_s = mu_s
        self.mu_i = mu_i

        self.x = np.linspace(0, L, M)
        self.X, self.Y = np.meshgrid(self.x, self.x)

        self.S = np.ones(M**2)
        self.I = np.zeros(M**2)
        self.R = np.zeros(M**2)

        self.I[self.X.ravel() < self.L * 0.2] = initial_infection
        self.I[self.X.ravel() < self.L * 0.1] = 0
        self.S -= self.I


        self.L_matrix = self.laplacian(M)
    
    def initial_condition(self, n):
        return

    
    def laplacian(self, M):
        Mi = M       # Number of inner points in each direction
        Mi2 = M**2  # Number of inner points in total

        # Construct a sparse A-matrix
        B = diags([1,-4,1],[-1,0,1],shape=(Mi, Mi), format="csr")
        A = kron(eye(Mi), B)
        C = diags([1,1],[-Mi,Mi],shape=(Mi2, Mi2), format="csr")
        A = (A+C).tocsr() # Konverter til csr-format (necessary for spsolve)
        plt.imshow((A/self.h).todense())
        plt.colorbar()
        plt.show()

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


    def show_initial(self):
        plt.imshow(self.I.reshape(self.M, self.M))
        plt.show()


    def animate(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        def update(frame):
            for _ in range(50): 
                self.step()
            ax.clear()
            ax.plot_surface(self.X, self.Y, self.I.reshape(self.M, self.M), cmap='hot', edgecolor='none')
            ax.set_zlim(0, 1)
            ax.set_title(f'Infected Population at t={frame*self.dt*50:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('Infected Fraction')

        anim = FuncAnimation(fig, update, frames=self.Nt//50, interval=10) 
        plt.show()

    

if __name__ == "__main__":
    sim = SIRSimulation()
    #sim.show_initial()
    sim.animate()