import numpy as np
import matplotlib.pyplot as plt

class DataGenerator:

    def __init__(self, added_noise: float, sampleSize: int):
        self.noise = added_noise
        self.sampleSize = sampleSize
        self.inputVector = None
        self.trueoutputVector = None
        self.noiseoutputVector = None
        self.training_noise_mean = None
        self.training_noise_std = None
        self.training_input_mean = None
        self.training_input_std = None
        self.inputVector_std = None
        self.noiseoutputVector_std = None


    def Generate_Non_Linear_Data(self, seed: int=None, standardize_data:bool=True, return_true_outputs: bool=False):

        '''
        Using function -> Sin(2*pi*x) + 0.5sin(10*pi*x) to generate data
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

        if standardize_data:

            std_input, std_noiseoutput = self.standardizeData()

            if return_true_outputs:
                return std_input, std_noiseoutput, self.trueoutputVector
            
            return std_input, std_noiseoutput

        
        if return_true_outputs:
            return self.inputVector, self.noiseoutputVector, self.trueoutputVector
        
        return self.inputVector, self.noiseoutputVector


    def standardizeData(self):

        '''This will standardize the training dataset'''
         
        if (self.inputVector is None) or (self.noiseoutputVector is None) or (self.trueoutputVector is None):
            raise ValueError("Data is not yet generated. Run .Generate_Non_Linear_Data()")
        
        # Standardizing input values

        input_mean = self.inputVector.mean(axis=0)
        input_std = self.inputVector.std(axis=0)

        if input_std == 0:
            input_std = 1e-6

        self.training_input_mean = input_mean
        self.training_input_std = input_std
        
        self.inputVector_std = (self.inputVector - input_mean)/input_std


        noise_mean = self.noiseoutputVector.mean(axis=0)
        noise_std = self.noiseoutputVector.std(axis=0)

        if noise_std == 0:
            noise_std = 1e-6

        # Saving noise mean and noise std to use on test_dataset
        self.training_noise_mean = noise_mean
        self.training_noise_std = noise_std

        self.noiseoutputVector_std = (self.noiseoutputVector - noise_mean)/noise_std

        return self.inputVector_std, self.noiseoutputVector_std

    
    def denormalize(self, predictions: np.ndarray):

        original_predictions = (predictions * self.training_noise_std) + self.training_noise_mean
        return original_predictions


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


