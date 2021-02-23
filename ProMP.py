from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.matlib as mat
from numpy.linalg import inv


def BasisFuncGauss(N, h, f, dt): 
    """
    Evaluates Gaussian basis functions in [0,1/f]
    N = number of basis functions
    h = bandwidth
    dt = time step
    f = modulation factor
    """
    tf = 1/f;
    T = int(round(tf/dt+1))
    Phi = np.zeros((T,N))
    for z in range(0,T):
        t = z*dt
        phi = np.zeros((1, N))
        for k in range(1,N+1):
            c = (k-1)/(N-1)
            phi[0,k-1] = np.exp(-(f*t - c)*(f*t - c)/(2*h))
        Phi[z,:N] = phi[0, :N]
    Phi = Phi/np.transpose(mat.repmat(np.sum(Phi,axis=1),N,1)); #[TxN]
    return Phi #[TxN]

class ProMP:
    """
    ProMP class
        N = number of basis functions
        h = bandwidth of basis functions
        dt = time step
        covQ = variance of original p(Q)
        Wm = mean of weights [Nx1]
        covW = variance of weights [NxN]
    internal:
        Phi = basis functions evaluated for every step [TxN]
        Qm = mean of Q [Nx1]
        cov = variance of p(Q|Wm) for every step  [Tx1]
    methods:
        condition (conditions an MP for a new viapoint)
            tstar = step of viapoint
            Qstar = value of Q of the viapoint
            covTstar = uncertainty of observation
        modulate (linear time modulation of the MP)
            factor = factor of linear modulation, z = factor*t
        printMP (plots an MP showing a standar deviation above and below)
            name = title of the plot
    """
    def __init__(self, N, h, dt, covQ, Wm, covW):
        self.N = N
        self.h = h
        self.dt = dt
        self.covQ = covQ
        self.Wm = Wm #[Nx1]
        self.covW = covW #[NxN]
        self.Phi = BasisFuncGauss(N,h,1,dt) #[TxN]
        self.T,_ = self.Phi.shape
        self.Qm = np.matmul(self.Phi,self.Wm) #[Tx1]
        self.cov = np.zeros((self.T,1)) #[Tx1]
        for i in range(0,self.T):
            self.cov[i,0] = self.covQ + np.matmul(np.array([self.Phi[i,:]]),np.matmul(covW,np.transpose(np.array([self.Phi[i,:]]))))
    def condition(self, tstar, Qstar, covQstar):
        Phit = np.transpose(self.Phi[tstar-1:tstar,:])
        self.Wm = # -- INSERT YOUR CODE HERE ---
        self.covW = # -- INSERT YOUR CODE HERE ---
        self.Qm = np.matmul(self.Phi,self.Wm)
        for i in range(0,self.T):
            self.cov[i,0] = self.covQ + np.matmul(np.array([self.Phi[i,:]]),np.matmul(self.covW,np.transpose(np.array([self.Phi[i,:]]))))
    def modulate(self, factor):
        self.Phi = BasisFuncGauss(self.N,self.h,factor,self.dt) #[TxN]
        self.T,_ = self.Phi.shape # new T
        self.Qm = np.matmul(self.Phi,self.Wm) #[Tx1]
        self.cov = np.zeros((self.T,1)) #[Tx1]
        for i in range(0,self.T):
            self.cov[i,0] = self.covQ + np.matmul(np.array([self.Phi[i,:]]),np.matmul(self.covW,np.transpose(np.array([self.Phi[i,:]]))))
    def printMP(self, name):
        t = np.arange(0, self.T*self.dt, self.dt)
        upper = # -- INSERT YOUR CODE HERE ---
        lower = # -- INSERT YOUR CODE HERE ---
        plt.plot(t,self.Qm)
        plt.fill_between(t, upper[:,0], lower[:,0], color = 'k', alpha = 0.1)
        plt.title(name)
        plt.show()

def blend(MP1,MP2,alpha1,alpha2):
    """
    blends two MPs
        MP1, MP2 = ProMP objects to blend
        alpha1, alpha2 = activation functions for each respective MP [Tx1]
    """
    a1 = np.transpose(np.array([alpha1])) #[Tx1]
    a2 = np.transpose(np.array([alpha2])) #[Tx1]
    cov12 = # -- INSERT YOUR CODE HERE ---
    Qm12 = # -- INSERT YOUR CODE HERE ---
    M12 = ProMP(MP1.N,MP1.h,MP1.dt,MP1.covQ,np.zeros((MP1.N,1)),np.zeros((MP1.N,MP1.N)))
    M12.cov = cov12
    M12.Qm = Qm12
    return M12


def main():
    # 15 weights obtained from 5 observations
    Wsamples = np.array([[0.0141,0.0130,0.0038,0.0029,0.0143],
                         [0.0044,0.2025,0.0178,0.0703,0.0143],
                         [0.0388,0.1042,0.0531,0.0854,0.1479],
                         [0.0025,0.0321,0.0235,0.0495,0.0086],
                         [0.0810,0.0178,0.1500,0.0310,0.0843],
                         [0.0658,0.1258,0.0488,0.1650,0.1398],
                         [0.1059,0.0821,0.0116,0.2260,0.0531],
                         [0.0032,0.0952,0.0305,0.2220,0.0025],
                         [0.2031,0.1665,0.1430,0.0842,0.0656],
                         [0.0491,0.1543,0.1232,0.1505,0.0049],
                         [0.1914,0.0525,0.0783,0.0009,0.0292],
                         [0.0584,0.1035,0.0830,0.0305,0.1452],
                         [0.0157,0.1713,0.2550,0.0695,0.0051],
                         [0.2106,0.0630,0.0942,0.0086,0.1512],
                         [0.0959,0.2093,0.1388,0.0566,0.0819]])

    Wmean = np.transpose([np.mean(Wsamples, axis=1)])
    Wcov = np.cov(Wsamples)
    N,_ = Wsamples.shape

    T = 100
    dt = 1/(T-1)

    # Define MPs
    MP1 = ProMP(N,0.02,dt,1e-6,Wmean,Wcov)
    MP2 = ProMP(N,0.02,dt,1e-6,Wmean,Wcov)
    MP3 = ProMP(N,0.02,dt,1e-6,Wmean,Wcov)

    # New desired point
    tstar1 = 100
    Qstar1 = MP2.Qm[100-1]+ 0.03
    covQstar1 = 1e-6
    MP2.condition(tstar1,Qstar1,covQstar1)

    t = np.arange(0, 1+dt, dt)

    # Plot original mean, and mean of conditioned MP
    plt.figure()
    plt.plot(t,MP1.Qm, 'r--', t, MP2.Qm, 'k')
    plt.plot(tstar1/(MP1.T), Qstar1,'ro')
    plt.title('Coditioning for point 1')
    plt.show()

    # Print MP conditioned to point 1
    MP2.printMP('Coditioning for point 1')

    # Second desired point
    tstar2 = 30
    Qstar2 = 0.11
    covQstar2 = 1e-6
    MP2.condition(tstar2,Qstar2,covQstar2)

    plt.figure()
    plt.plot(t,MP1.Qm, 'r--', t, MP2.Qm, 'k')
    plt.plot(tstar1/(MP1.T), Qstar1,'ro')
    plt.plot(tstar2/(MP1.T), Qstar2,'ro')
    plt.title('Coditioning for point 1 and point 2')
    plt.show()

    MP2.printMP('Coditioning for point 1 and point 2')

    # Blending: MP1 is conditioned to point 1, MP2 is conditioned to point 2, MP12 is the result of blending both
    MP1 = ProMP(N,0.02,dt,1e-6,Wmean,Wcov)
    MP2 = ProMP(N,0.02,dt,1e-6,Wmean,Wcov)
    MP1.condition(tstar1,Qstar1,covQstar1)
    MP2.condition(tstar2,Qstar2,covQstar2)
    MP1.printMP('MP1: Coditioning for point 1')
    MP2.printMP('MP2: Coditioning for point 2')
    # Activation functions for blending
    alpha1 = # -- INSERT YOUR CODE HERE --
    alpha2 = -alpha1+1
    MP12 = blend(MP1,MP2,alpha1,alpha2)
    MP12.printMP('blending of MP1 and MP2')

    #time modulation
    MP1 = ProMP(N,0.02,dt,1e-6,Wmean,Wcov)
    MP2 = ProMP(N,0.02,dt,1e-6,Wmean,Wcov)
    MP3 = ProMP(N,0.02,dt,1e-6,Wmean,Wcov)
    MP2.modulate(0.75)
    MP3.modulate(1.5)

    t1 = np.arange(0, MP1.T*dt, dt)
    t2 = np.arange(0, MP2.T*dt, dt)
    t3 = np.arange(0, MP3.T*dt, dt)

    # Plot original mean, and mean of conditioned MP
    plt.figure()
    plt.plot(t1,MP1.Qm, 'k')
    plt.plot(t2,MP2.Qm, 'b')
    plt.plot(t3,MP3.Qm, 'g')
    plt.title('Time modulation')
    plt.show()

    MP1.printMP('modulation factor = 1')
    MP2.printMP('modulation factor = 0.75')
    MP3.printMP('modulation factor = 1.5')


if __name__ == "__main__":
    main()
