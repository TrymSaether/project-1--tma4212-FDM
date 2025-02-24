import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.sparse import kron
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class SIR:
    def __init__(self, M=50, dt=0.001, T=0.1, beta=3.0, mu_s=0.1, mu_i=0.5):
        self.M = M
        self.dt = dt
        self.nsteps = int(T / dt)
        self.beta = beta
        self.mu_s = mu_s
        self.mu_i = mu_i
        self.h = 1.0 / M
        self.L = self._build_laplacian(M)
        self.L *= 1.0 / (self.h**2)
        self.S = np.ones((M, M)).ravel()
        self.I = np.zeros((M, M)).ravel()
        self.I[:5*5] = 0.2

    def _build_laplacian(self, m):
        c = diags([-1,2,-1], [-1,0,1], shape=(m,m))
        I_m = eye(m)
        return kron(c, I_m) + kron(I_m, c)

    def _step(self):
        S2d = self.S.reshape(self.M, self.M)
        I2d = self.I.reshape(self.M, self.M)
        SI = self.beta * S2d * I2d
        dS = -SI.ravel() + self.mu_s * self.L.dot(self.S)
        dI =  SI.ravel() + self.mu_i * self.L.dot(self.I)
        self.S += self.dt * dS
        self.I += self.dt * dI

    def solve(self):
        for _ in range(self.nsteps):
            self._step()
        return self.S, self.I
    
if __name__ == "__main__":
    solver = SIR(M=50, dt=0.0005, T=0.05, beta=3.0, mu_s=0.1, mu_i=0.5)
    S_final, I_final = solver.solve()

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    M = solver.M
    frames = 30
    skip = solver.nsteps // frames if solver.nsteps > frames else 1
    S_list, I_list = [], []
    solver2 = SIR(M=50, dt=0.0005, T=0.05, beta=3.0, mu_s=0.1, mu_i=0.5)
    for n in range(solver2.nsteps+1):
        if n % skip == 0:
            S_list.append(solver2.S.copy())
            I_list.append(solver2.I.copy())
        solver2._step()

    fig, ax = plt.subplots()
    im = ax.imshow(I_list[0].reshape(M,M), origin='lower', cmap='hot',
                   vmin=0, vmax=np.max(I_list[0]),
                   extent=[0,1,0,1])
    fig.colorbar(im, ax=ax)
    ax.set_title("I(t=0)")

    def update(frame):
        arr = I_list[frame].reshape(M, M)
        im.set_data(arr)
        im.set_clim(0, arr.max() + 1e-9)
        ax.set_title(f"I, frame={frame}")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(I_list), interval=200, blit=False)
    plt.show()


