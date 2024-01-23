import numpy as np
import tqdm

from scipy import sparse
import torch

def createNDimensionalMeshGrid(numDimensions: int, pointsPerDimension: int, dimensionLimits: tuple[int, int]):
    """
    Create the mesh grid representing the discretized space to operate over.

    Note that numDimensions and pointsPerDimension will increase the computational complexity exponentially!!
    """

    return np.meshgrid(*[np.linspace(dimensionLimits[0], dimensionLimits[1], pointsPerDimension) for _ in range(numDimensions)])


def getSecondPartialMatrixApproximation(pointsPerDimension: int):
    """
    Get the discretized matrix approximation of the second partial derivate, del^2/del x^2
    """
    onesArray = np.ones([pointsPerDimension])
    diagonalEntries = np.array([onesArray, -2*onesArray, onesArray])
    return sparse.spdiags(diagonalEntries, np.array([-1,0,1]), pointsPerDimension, pointsPerDimension)


def getKineticSchrodingerTerm(numDimensions: int, pointsPerDimension: int):
    """
    Get the kinetic energy term for the Schrodinger equation, 
    effectively just a second partial derivate matrix for each dimension, kronecker summed together.
    """
    D = getSecondPartialMatrixApproximation(pointsPerDimension)
    finalMatrix = D

    for _ in range(numDimensions-1):
        finalMatrix = sparse.kronsum(finalMatrix, D)
    
    return -1/2 * finalMatrix

class TorchTimeIndependentSchrodingerSolver():

    def __init__(self, hamiltonian, pointsPerDimension: int):
        self.pointsPerDimension = pointsPerDimension
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            print("WARNING: torch device set to CPU. Check CUDA installation.")

        hamiltonian = hamiltonian.tocoo()
        self.hamiltonianTensor = torch.sparse_coo_tensor(
            indices=torch.tensor([hamiltonian.row, hamiltonian.col]), 
            values=torch.tensor(hamiltonian.data), 
            size=hamiltonian.shape
        ).to(self.device)

        self.numEigenstates = 0

    def solveEigenstates(self, numEigenstates: int):
        self.numEigenstates = numEigenstates
        self.eigenvalues, self.eigenvectors = torch.lobpcg(self.hamiltonianTensor, k=numEigenstates, largest=False)

    def getEigenstate(self, eigenstateIndex: int):
        if self.numEigenstates == 0:
            print("Cannot get eigenstates before solveEigenstates is called")
        if self.numEigenstates < eigenstateIndex:
            print("Cannot get eigenstate index larger than numEigenstates solved for")

        return (
            self.eigenvalues[eigenstateIndex], 
            self.eigenvectors.T[eigenstateIndex]
                .reshape(self.pointsPerDimension, self.pointsPerDimension)
                .cpu()
            )
    
class TimeDependentSchrodingerSolver():

    def __init__(self, potential, discreteSpatialMeshgrid,  discreteTemporalSpacing: float):
        """
        Creates a new Time Dependent Schrodinger Solver using the Finite Difference method.

        Assumes potential is infinite at edges.
        """
        
        self.potential = potential
        
        self.discreteSpatialMeshgrid = discreteSpatialMeshgrid
        self.numDimensions = len(discreteSpatialMeshgrid)
        self.pointsPerDimension = len(discreteSpatialMeshgrid[0])
        self.dimensionalLimits = (np.min(discreteSpatialMeshgrid[0]), np.max(discreteSpatialMeshgrid[0]))

        self.discreteSpatialSpacing = 1 / (self.pointsPerDimension - 1)
        self.discreteTemporalSpacing = discreteTemporalSpacing

    def getDelTimeOverDelXSquare(self) -> float:
        """
        This value acts as a measure of how good the finite difference method will hold. Smaller is better!
        """

        return self.discreteTemporalSpacing / self.discreteSpatialSpacing**2
    
    def normalizePsi(self, psi):
        normalizationFactor = np.sum(np.abs(psi)**2)*self.discreteSpatialSpacing
        return psi / normalizationFactor

    def solve(self, initialPsi, numTimesteps: int):
        spatialDimensions = [self.pointsPerDimension for _ in range(self.numDimensions)]
        psi = np.zeros([numTimesteps, *spatialDimensions])
        psi[0] = initialPsi

        return self._compute_psi(psi.astype(complex), numTimesteps)

    def _compute_psi(self, psi, numTimesteps: int):
        for t in tqdm.tqdm(range(1, numTimesteps), "Timestep: "):
            for i in range(1, self.pointsPerDimension-1):
                psi[t][i] = psi[t-1][i] + 1j/2 * self.getDelTimeOverDelXSquare() * (psi[t-1][i+1] - 2*psi[t-1][i] + psi[t-1][i-1]) - 1j * self.discreteTemporalSpacing * self.potential[i]*psi[t-1][i]
            psi[t] = self.normalizePsi(psi[t])
        return psi