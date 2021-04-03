CHUNK = 2048  # audio frame length
STEP = 350.0  # 每个频率的跨度
NUM_OF_FREQ = 8  # 频率数量
DELAY_TIME = 1  # 麦克风的延迟时间
STD_THRESHOLD = 0.022  # 相位标准差阈值
N_CHANNELS = 7  # 声道数
F0 = 17000.0

PADDING_LEN = 1400

data_shape = (NUM_OF_FREQ * N_CHANNELS, PADDING_LEN)

phase_input_shape = (NUM_OF_FREQ * N_CHANNELS, PADDING_LEN, 1)

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