import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.Dense(10)
])

# model.summary()
# plot_model(model,to_file="model2.png",show_shapes=True)
l1 = keras.layers.Bidirectional(keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]))
l2 = keras.layers.SimpleRNN(30, return_sequences=True)
l3 = keras.layers.Dense(10)
import numpy as np

input_x: np.ndarray = np.random.random(1000).reshape(10, 1, 100)
x1 = l1(input_x.astype(np.float32))
x2 = l2(x1)
x3 = l3(x2)
pass
