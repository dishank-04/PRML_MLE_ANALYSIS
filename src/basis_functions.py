import numpy as np

class PolynomialBasis:

    def __init__(self, degree):
        self.degree = degree


    def transform(self, input_values: np.ndarray):

        '''
        This will create Design Matrix, Now we will take every single value and take it to the degree value

        Please input the vector with ndim = 2, a column Vector
        '''

        sampleSize = input_values.shape[0]

        if input_values.ndim < 2: 
            raise ValueError('Please make input_values a column vector of (N,1)')
        
        powers = np.arange(self.degree+1) # This will create [0, 1, ... self.degree+1] powers. Power 0 will create Bias column of all 1's.

        design_matrix = np.power(input_values, powers)
        return design_matrix
    


