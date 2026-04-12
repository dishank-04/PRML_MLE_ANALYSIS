from src.data_generator import DataGenerator
from src.models import LinearModel

data = DataGenerator(0.4, 10)
X,t = data.Generate_Non_Linear_Data()
print(X)
print(t)
