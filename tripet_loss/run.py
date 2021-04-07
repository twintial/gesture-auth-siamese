from config import phase_input_shape
from siamese_cons_loss.cnn import cons_cnn_model
from tripet_loss.model import TripLossModel


def main():
    input_shape = phase_input_shape
    cnn_net = cons_cnn_model(input_shape)
    model = TripLossModel(cnn_net, input_shape, 30, 30, 0.3)
    model.model.summary()


if __name__ == '__main__':
    main()
