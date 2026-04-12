from src.basis_functions import PolynomialBasis
import numpy as np

class LinearModel:

    def __init__(self, basis_func_degree):
        self.weights = None
        self.basisFuncDegree = basis_func_degree
    
    def fitNormalEquation(self, X_training: np.ndarray, t_training: np.ndarray):

        basis = PolynomialBasis(self.basisFuncDegree)
        phi = basis.transform(X_training) # This will create Design Matrix of N x M

        A = phi.T @ phi
        b = phi.T @ t_training

        self.weights = np.linalg.solve(A, b) # Not using np.linalg.inv to solve for optimal weights as its computationally expensive O(N^3)

    
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

        if self.weights is None:
            raise ValueError('Model is not yet fit. Fit model using .fitNormalEquation() or .fitGradientDescent()')

        basis = PolynomialBasis(self.basisFuncDegree)
        phi = basis.transform(test_input)

        predictions = phi @ self.weights

        return predictions
    

    def ERMS(self, test_output: np.ndarray, test_input: np.ndarray):
        
        predictions = self.predict(test_input)
        
        # Mean Squared Error
        mse = np.mean(np.square(test_output - predictions))
        
        # Root Mean Squared Error
        erms = np.sqrt(mse)
        
        return erms
