from src.data_generator import DataGenerator
from src.basis_functions import PolynomialBasis
from src.models import LinearModel

data = DataGenerator(0.4, 5)
X,t = data.Generate_Non_Linear_Data()
model = LinearModel(3)
model.fit(X,t)
print(model.weights)