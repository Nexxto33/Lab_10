import numpy as np
import tensorflow as tf
from tensorflow import keras

X_train = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
Y_train = 3 * X_train + 2  # Y = 3X + 2

model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(X_train, Y_train, epochs=500, verbose=0)

X_test = np.array([5.0, 3.3])
Y_pred = model.predict(X_test)

print("Predicciones:")
for i, x in enumerate(X_test):
    print(f"Para X = {x:.1f}, el modelo predice Y ≈ {Y_pred[i][0]:.2f}")
