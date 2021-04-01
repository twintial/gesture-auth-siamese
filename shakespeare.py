import os

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

DEBUG = False

shakespeare_url = "https://homl.info/shakespeare" # shortcut URL
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([shakespeare_text])

if DEBUG:
    print(tokenizer.texts_to_sequences(["First"]))
    print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))
max_id = len(tokenizer.word_index)  # number of distinct characters


[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
dataset_size = len(encoded)
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
# window
n_steps = 100
window_length = n_steps + 1  # target = input 向前移动1个字符
dataset = dataset.window(window_length, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))

batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))

dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))

dataset = dataset.prefetch(1)


model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                     dropout=0.2, recurrent_dropout=0.2),
    keras.layers.GRU(128, return_sequences=True,
                     dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                    activation="softmax"))
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam")
history = model.fit(dataset, epochs=20)

model_version = "0001"
model_name = "my_shakespeare_model"
model_path = os.path.join(model_name, model_version)
tf.saved_model.save(model, model_path)
