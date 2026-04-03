from src.data_generator import DataGenerator

data = DataGenerator(0.4, 80)
X,t = data.Generate_Non_Linear_Data()
print(X.shape)
print(t.shape)
