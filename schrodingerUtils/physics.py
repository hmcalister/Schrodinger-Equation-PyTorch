import numpy as np

from scipy import sparse
import torch

def createNDimensionalMeshGrid(numDimensions: int, pointsPerDimensions: int, dimensionLimits: tuple[int, int]):
    """
    Create the mesh grid representing the discretized space to operate over.

    Note that numDimensions and pointsPerDimension will increase the computational complexity exponentially!!
    """

    return np.meshgrid(*[np.linspace(dimensionLimits[0], dimensionLimits[1], pointsPerDimensions) for _ in range(numDimensions)])


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

class TorchSchrodingerSolver():

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