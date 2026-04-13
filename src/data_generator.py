import numpy as np
import matplotlib.pyplot as plt

class DataGenerator:

    def __init__(self, added_noise: float, sampleSize: int):
        self.noise = added_noise
        self.sampleSize = sampleSize
        self.inputVector = None
        self.trueoutputVector = None
        self.noiseoutputVector = None

    def standardizeData():
        pass

    def Generate_Non_Linear_Data(self, seed: int=None):

        ''' For analysis part we will be using Composite Sin function as our true function which is: sin*(2*pi*x) + 0.5*sin(10*pi*x)

        This will have only 1 single Feature which is input_values
        '''

        input_values = np.linspace(0, 1, self.sampleSize)    
        true_output_values = np.sin(2*np.pi*input_values) + 0.5*np.sin(10*np.pi*input_values) # THis create a column vector

        rng = np.random.default_rng(seed)

        noise = rng.normal(0, self.noise, self.sampleSize)

        noise_added_outputs = true_output_values + noise
    
        input_values_vector = input_values.reshape(-1,1)
        noise_added_outputs_vector = noise_added_outputs.reshape(-1,1)

        self.inputVector = input_values_vector
        self.noiseoutputVector = noise_added_outputs_vector
        self.trueoutputVector = true_output_values.reshape(-1,1)

        return input_values_vector, noise_added_outputs_vector, self.trueoutputVector
    
    def Generate_Linear_Data(self):
        pass


    def plot(self,show_true_function=False):

        '''
        This will plot the DataPoints
        '''

        if self.inputVector is None:
                raise ValueError("Data has not been generated yet. Call Generate_Non_Linear_Data first.")

        
        plt.scatter(self.inputVector, self.noiseoutputVector, color='navy', label='Data Points')
                
        if show_true_function:
            plt.plot(self.inputVector, self.trueoutputVector, color='red', label='True Function')
            plt.title('Function: sin(2*pi*x) + 0.5*sin(10*pi*x)')
            
        plt.xlabel('X_values')
        plt.ylabel('t_values')
        plt.legend()
        plt.show()

        






        













