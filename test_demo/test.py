# import tensorflow as tf
import numpy as np

# a = np.array([[t,2],[2,3]])
# b = tf.constant(a)
# c = tf.io.serialize_tensor(b)
#
# x = tf.train.BytesList(value=[c.numpy()])
#
# d = tf.io.parse_tensor(c.numpy(), out_type=tf.int32)
pass
# dataset = tf.data.Dataset.range(10) # 0 to 9, three times
# dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=20, count=t, seed=42)).batch(7)
# for item in dataset:
#      print(item)

# size = 3
# a = np.array([[t, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8]])
# a_l = a.shape[t]
# delta_len = size - a_l
#
# left_clip_len = abs(delta_len) // 2
# right_clip_len = abs(delta_len) - left_clip_len
# print(a[:, left_clip_len:-right_clip_len])

#
# left_zero_padding_len = abs(delta_len) // 2
# right_zero_padding_len = abs(delta_len) - left_zero_padding_len
# print(np.pad(a, ((0, 0), (left_zero_padding_len, right_zero_padding_len))))
pass
import tensorflow as tf
a = tf.constant([1,2,3,4])
b = tf.reshape(a, (2,2))
print(b)
print(tf.reduce_sum(b, axis=1))

