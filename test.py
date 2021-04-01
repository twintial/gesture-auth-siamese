import tensorflow as tf
import numpy as np
a = np.array([[1,2],[2,3]])
b = tf.constant(a)
c = tf.io.serialize_tensor(b)

x = tf.train.BytesList(value=[c.numpy()])

d = tf.io.parse_tensor(c.numpy(), out_type=tf.int32)
pass