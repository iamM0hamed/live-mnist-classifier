import keras
from keras import layers


def build_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPool2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPool2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
