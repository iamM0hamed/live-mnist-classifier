from keras.datasets import mnist
from keras.saving import load_model

from preprocessing import preprocess_images

(_, _), (x_test, y_test) = mnist.load_data()

x_test = preprocess_images(x_test)

model = load_model("models/mnist_cnn.keras")

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
