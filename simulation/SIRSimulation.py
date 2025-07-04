import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, eye, kron

class SIRSimulation:
    def __init__(self, L=1.0, M=50, T=10, dt=0.001, initial_infection=0.01,
                 beta=3, gamma=1, mu_s=0.001, mu_i=0.001, n=0, dynamic_beta=False):
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
        self.db = dynamic_beta

        self.x = np.linspace(0, L, M)
        self.X, self.Y = np.meshgrid(self.x, self.x)

        self.S = np.ones(M**2)
        self.I = np.zeros(M**2)
        self.R = np.zeros(M**2)

        self.initial_condition()
        self.S -= self.I
        
        self.L_matrix = self.laplacian()
    
    def initial_condition(self):
        if self.n == 0:  #Dense infection in corners
            dist_to_corner = (self.X - self.L / 4)**2 + (self.Y - self.L / 4)**2
            infection = (self.ii * np.exp(-200 * dist_to_corner)).ravel()
            self.I = infection + infection[::-1]
            
        elif self.n == 1:  #Linear front of infection
            self.I[self.X.ravel() < self.L * 0.2] = self.ii

        else:  #3 random infection areas
            cx, cy = np.random.uniform(0.3, 0.7, 3) * self.L, np.random.uniform(0.1, 0.9, 3) * self.L
            X_flat, Y_flat = self.X.ravel(), self.Y.ravel()
            distances = np.min((X_flat[:, None] - cx) ** 2 + (Y_flat[:, None] - cy) ** 2, axis=1)
            self.I[distances < 0.009] = self.ii



    def laplacian(self):
        diag = -2 * np.ones(self.M)
        diag[0] = diag[-1] = -1
        off = np.ones(self.M - 1)

        L = diags([off, diag, off], [-1, 0, 1], shape=(self.M, self.M), format='csr') / self.h**2
        I = eye(self.M, format='csr')
        
        return kron(I, L) + kron(L, I)

    def beta_function(self, t):
        if 2 <= t <= 2+np.pi:
            time_factor = 1 + 0.1 * np.sin(t-2)  # Fluctuates with time
        else:
            return self.beta
        
        dist_to_center = (self.X - self.L / 2)**2 + (self.Y - self.L / 2)**2
        spatial_factor = 1 + 0.5 * np.exp(-100 * dist_to_center)  # Higher in center

        return (self.beta * spatial_factor * time_factor).ravel()


    def step(self, t):
        if self.db:
            beta = self.beta_function(t)
        else:
            beta = self.beta

        infection = beta * self.S * self.I
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
        surf = [None]
        
        def update(frame):
            t = frame * self.dt * 50  # Compute current time
            if t >= 10:  # Stop condition
                anim.event_source.stop()
                return
            
            for _ in range(50): 
                self.step(t)  # Stops updating once t > 10
            
            if surf[0] is not None:
                surf[0].remove()
            
            surf[0] = ax.plot_surface(self.X, self.Y, self.I.reshape(self.M, self.M), cmap='hot', edgecolor='none')
            ax.set_zlim(0, 1)
            ax.set_title(f'Infected Population at t={t:.2f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('Infected Fraction')
        
        anim = FuncAnimation(fig, update, frames=int(min(self.Nt//50, 10 / (self.dt * 50))), interval=10) 
        plt.show()


sim = SIRSimulation(n=0, dynamic_beta=True)
sim.show_initial()
sim.animate()