import numpy as np
from keras.datasets import mnist
from keras.saving import load_model

from preprocessing import preprocess_images

(_, _), (x_test, y_test) = mnist.load_data()
x_test = preprocess_images(x_test)

model = load_model("models/mnist_cnn.keras")

image = x_test[0]
image = np.expand_dims(image, axis=0)

predictions = model.predict(image)
predicted_digit = np.argmax(predictions[0])

print("Predicted:", predicted_digit)
print("Actual:", y_test[0])

print("Probabilities:")

for digit, probability in enumerate(predictions[0]):
    print(f"{digit}: {probability:.4f}")
