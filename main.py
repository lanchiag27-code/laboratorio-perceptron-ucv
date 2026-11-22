from src.perceptron import Perceptron
from src.dataset import training_data

model = Perceptron()
model.train(training_data)

prediction = model.predict([9, 1])

print("Resultado de la prediccion:", prediction)
print("Pesos finales:", model.weights)
print("Bias final:", model.bias)
