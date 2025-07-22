"""
Train a CNN on MNIST and save the model.
Run:  python mnist_cnn.py
"""

import os, pathlib, datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# ---------- 1. Load & preprocess data ----------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0            # scale to [0,1]
x_test  = x_test.astype("float32")  / 255.0
x_train = np.expand_dims(x_train, -1)                  # (N, 28, 28, 1)
x_test  = np.expand_dims(x_test,  -1)
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# ---------- 2. Build CNN ----------
model = models.Sequential([
    layers.Conv2D(32, 3, activation="relu", input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# ---------- 3. Train ----------
EPOCHS   = 12
BATCH_SZ = 128
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SZ
)

# ---------- 4. Evaluate ----------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅  Test accuracy: {test_acc*100:.2f}%")

# ---------- 5. Save ----------
out_dir = pathlib.Path("saved_model")
out_dir.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
model.save(out_dir / f"mnist_cnn_{timestamp}.keras")

