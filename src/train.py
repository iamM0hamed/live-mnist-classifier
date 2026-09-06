from keras.datasets import mnist

from model import build_model
from preprocessing import preprocess_images

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)

model = build_model()

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(test_accuracy)

model.save("models/mnist_cnn.keras")
