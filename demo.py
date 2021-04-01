import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model, layers
from tensorflow.keras.utils import plot_model

model = keras.models.Sequential([
    layers.Input(shape=(None, 8, 777, 1)),
    layers.TimeDistributed(layers.Conv2D(filters=16, kernel_size=(2, 8), padding='same'))
])

model.summary()
plot_model(model,to_file="model2.png",show_shapes=True)
