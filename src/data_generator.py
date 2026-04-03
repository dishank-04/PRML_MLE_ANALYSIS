import numpy as np
import matplotlib.pyplot as plt

class DataGenerator:

    def __init__(self, added_noise: float, sampleSize: int):
        self.noise = added_noise
        self.sampleSize = sampleSize
        self.inputVector = None
        self.trueoutputVector = None
        self.noiseoutputVector = None

    def Generate_Non_Linear_Data(self):

        ''' For analysis part we will be using Composite Sin function as our true function which is: sin*(2*pi*x) + 0.5*sin(10*pi*x)

        This will have only 1 single Feature which is input_values
        '''

        input_values = np.linspace(0, 1, self.sampleSize)    
        true_output_values = np.sin(2*np.pi*input_values) + 0.5*np.sin(10*np.pi*input_values)

        
        np.random.seed(self.sampleSize) # This will help to keep our noise constant without changing the numbers everytime we run the code, helps with reproducability of data.

        noise = np.random.normal(0, self.noise, self.sampleSize)

        noise_added_outputs = true_output_values + noise
    
        input_values_vector = input_values.reshape(-1,1)
        noise_added_outputs_vector = noise_added_outputs.reshape(-1,1)

        self.inputVector = input_values_vector
        self.noiseoutputVector = noise_added_outputs_vector
        self.trueoutputVector = true_output_values.reshape(-1,1)

        return input_values_vector, noise_added_outputs_vector
    
    def Generate_Linear_Data(self):
        pass


    def plot(self,show_true_function=False):

        '''
        This will plot the DataPoints
        '''

        if show_true_function == True:

            plt.scatter(self.inputVector, self.noiseoutputVector, color='navy', label='Data_Points')
            plt.plot(self.inputVector, self.trueoutputVector, color='red', label='true_Function')
            plt.xlabel('X_values')
            plt.ylabel('t_values')
            plt.title('Function is: sin*(2*pi*x) + 0.5*sin(10*pi*x)')
            plt.legend()
            plt.show()
        
        else:
                
            plt.scatter(self.inputVector, self.noiseoutputVector, color='navy', label='Data_Points')
            plt.xlabel('X_values')
            plt.ylabel('t_values')
            plt.legend()
            plt.show()

        






        













