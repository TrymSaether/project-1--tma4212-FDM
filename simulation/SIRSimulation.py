import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, eye, kron

class SIRSimulation:
    def __init__(self, n=0, L=1.0, M=50, T=10, dt=0.001, initial_infection=0.01,
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
        self.ii = initial_infection
        self.n = n

        self.x = np.linspace(0, L, M)
        self.X, self.Y = np.meshgrid(self.x, self.x)

        self.S = np.ones(M**2)
        self.I = np.zeros(M**2)
        self.R = np.zeros(M**2)

        self.initial_condition()
        self.S -= self.I
        
        self.L_matrix = self.laplacian()
    
    def initial_condition(self):
        if self.n == 0:  # 3 random infection areas
            cx, cy = np.random.uniform(0.1, 0.9, 3) * self.L, np.random.uniform(0.1, 0.9, 3) * self.L
            X_flat, Y_flat = self.X.ravel(), self.Y.ravel()
            distances = np.min((X_flat[:, None] - cx) ** 2 + (Y_flat[:, None] - cy) ** 2, axis=1)
            self.I[distances < 0.009] = self.ii

        elif self.n == 1:  # Linear front of infection
            self.I[self.X.ravel() < self.L * 0.2] = self.ii

        else:  # Dense central infection
            dist_to_center = (self.X - self.L / 2) ** 2 + (self.Y - self.L / 2) ** 2
            self.I = self.ii * np.exp(-100 * dist_to_center)




    def laplacian(self):
        diag = -2 * np.ones(self.M)
        diag[0] = diag[-1] = -1
        off = np.ones(self.M - 1)

        L = diags([off, diag, off], [-1, 0, 1], shape=(self.M, self.M), format='csr') / self.h**2
        I = eye(self.M, format='csr')
        
        return kron(I, L) + kron(L, I)

    

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


sim = SIRSimulation()
sim.show_initial()
