import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron

class SIRSimulation:
    def __init__(self, L=10.0, Nx=50, Ny=50, T=10, dt=0.01, initial_infection=0.1):
        self.L = L
        self.Nx, self.Ny = Nx, Ny
        self.h = L / Nx
        self.T = T
        self.dt = dt
        self.Nt = int(T / dt)
        self.initial_infection = initial_infection
        
        self.x = np.linspace(0, L, Nx)
        self.y = np.linspace(0, L, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.S = np.ones((Nx, Ny))
        self.I = np.zeros((Nx, Ny))
        self.R = np.zeros((Nx, Ny))
        
        self.center_x, self.center_y = Nx // 2, Ny // 2
        self.I[self.center_x-2:self.center_x+2, self.center_y-2:self.center_y+2] = self.initial_infection
        self.S -= self.I


    def laplacian_1d(self, size):
        diagonals = [-np.ones(size - 1), 2 * np.ones(size), -np.ones(size - 1)]
        L = diags(diagonals, [-1, 0, 1], format="csr")
        
        L.data[L.indptr[0]:L.indptr[1]] = 0
        L.data[L.indptr[-2]:L.indptr[-1]] = 0

        return L

    def laplacian_2d(self, n, m):
        C_m = self.laplacian_1d(m)
        C_n = self.laplacian_1d(n)

        L2D = kron(C_n, eye(m, format="csr")) + kron(eye(n, format="csr"), C_m)

        return L2D
    
    def vic(self):
        plt.figure(figsize=(6, 5))
        plt.imshow(self.I, cmap='hot', origin='lower', extent=[0, self.L, 0, self.L])
        plt.colorbar(label='Initial Infected Fraction')
        plt.title('Initial Infection Spread')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


sim = SIRSimulation()
sim.vic()


