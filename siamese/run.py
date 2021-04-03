from tensorflow.python.keras.utils.vis_utils import plot_model

from config import *
from siamese.cnn import cons_cnn_model
from siamese.siamese_net import SiameseNet


def main():
    # 构建siamese
    input_shape = phase_input_shape
    cnn_net = cons_cnn_model(input_shape)
    siamese_net = SiameseNet(cnn_net)
    siamese_net.model.summary()
    plot_model(siamese_net.model, to_file="siamese.png", show_shapes=True)


if __name__ == '__main__':
    main()
