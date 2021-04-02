import tensorflow.keras as keras
from tensorflow.keras.utils import plot_model
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.Dense(10)
])

model.summary()
plot_model(model,to_file="model2.png",show_shapes=True)