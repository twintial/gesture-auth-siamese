import tensorflow as tf
filename = r'D:\projects\pyprojects\gesturerecord\location\1khz\0.wav'
audio_binary = tf.io.read_file(filename)
data, fs = tf.audio.decode_wav(audio_binary)  # 会变成-t，t
