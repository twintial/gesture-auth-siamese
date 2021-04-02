import tensorflow as tf
import numpy as np
# a = np.array([[1,2],[2,3]])
# b = tf.constant(a)
# c = tf.io.serialize_tensor(b)
#
# x = tf.train.BytesList(value=[c.numpy()])
#
# d = tf.io.parse_tensor(c.numpy(), out_type=tf.int32)
pass
dataset = tf.data.Dataset.range(10) # 0 to 9, three times
dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=20, count=1, seed=42)).batch(7)
for item in dataset:
     print(item)