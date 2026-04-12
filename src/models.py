from src.basis_functions import PolynomialBasis
import numpy as np

class LinearModel:

    def __init__(self, basis_func_degree):
        self.weights = None
        self.basisFuncDegree = basis_func_degree
    
    def fitNormalEquation(self, X_training: np.ndarray, t_training: np.ndarray):

        basis = PolynomialBasis(self.basisFuncDegree)
        phi = basis.transform(X_training) # This will create Design Matrix of N x M


        try:
            covariance_matrix = phi.T @ phi
            weights = (np.linalg.inv(covariance_matrix)) @ phi.T @ t_training

            self.weights = weights

        except np.linalg.LinAlgError:
            
            print("Matrix is Singular falling back to Pseduo-inverse")

            covariance_matrix = phi.T @ phi
            weights = (np.linalg.pinv(covariance_matrix)) @ phi.T @ t_training

            self.weights = weights
    
    def fitGradientDescent(self, X_training: np.ndarray, t_training:np.ndarray, learning_rate: float, epochs: int):

        sampleSize = X_training.shape[0]

        basis = PolynomialBasis(self.basisFuncDegree)
        phi = basis.transform(X_training) # This will create Design Matrix

        weights = np.random.randn(phi.shape[1],1)

        for i in range(epochs):

            predictions = phi @ weights

            gradient = ((phi.T @ (predictions - t_training)))/sampleSize # Dividing Gradient by number of samples so learning rate doesnt have to be too small to prevent Explosions. 

            if i%10 == 0:
                mse = 1/(2*sampleSize)*np.sum((predictions - t_training)**2)
                print(mse)

            weights = weights - learning_rate*gradient

        self.weights = weights


    def predict(self, test_input: np.ndarray):

        if self.weight is None:
            raise ValueError('Model is not yet fit. Fit model using .fitNormalEquation() or .fitGradientDescent()')

        basis = PolynomialBasis(self.basisFuncDegree)
        phi = basis.transform(test_input)

        predictions = phi @ self.weights

        return predictions



        



