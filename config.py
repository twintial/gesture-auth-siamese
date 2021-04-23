CHUNK = 2048  # audio frame length
FS = 48000  # sample rate
NUM_OF_FREQ = 8  # 频率数量
DELAY_TIME = 1  # 麦克风的延迟时间
STD_THRESHOLD = 0.022  # 相位标准差阈值,没用
N_CHANNELS = 7  # 声道数

'''
这两个会修改。
第一次用的是:
F0 = 17000.0
STEP = 350.0
第二次用的是:
F0 = 17350.0
STEP = 700.0
'''
F0 = 17350.0
STEP = 700.0  # 每个频率的跨度


I_Q_skip = 1000

PADDING_LEN = 1400

# 数据集转换线程数
num_parallel_calls = 2

# neural network config
TF_CPP_MIN_LOG_LEVEL = '1'
data_shape = (NUM_OF_FREQ * N_CHANNELS, PADDING_LEN)
phase_input_shape = (NUM_OF_FREQ * N_CHANNELS, PADDING_LEN, 1)
BATCH_SIZE = 32
# 与数据集大小有关，数据集越大这个值越大
SHUFFLE_BUFF_SIZE = 10000
RANDOM_SEED = 12

# train
TRAINING_AUDIO_DIRS = [
    # r'D:\projects\pyprojects\soundphase\gest\sjj\gesture1',
    # r'D:\projects\pyprojects\soundphase\gest\sjj\gesture2',
    # r'D:\projects\pyprojects\soundphase\gest\sjj\gesture3'
    # r'D:\实验数据\2021\毕设\micarray\sjj\gesture1',
    # r'D:\实验数据\2021\毕设\micarray\sjj\gesture2',
    # r'D:\实验数据\2021\毕设\micarray\sjj\gesture3'
    r'D:\实验数据\2021\毕设\micarrayspeaker\sjj\gesture2',
    # r'D:\实验数据\2021\毕设\micarrayspeaker\sjj\gesture3',
    # r'D:\实验数据\2021\毕设\micarrayspeaker\sjj\gesture4',
    # r'D:\实验数据\2021\毕设\micarrayspeaker\sjj\gesture5',
    # r'D:\实验数据\2021\毕设\distant_mic\sjj\gesture1',
]
