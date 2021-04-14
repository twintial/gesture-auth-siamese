import tensorflow as tf
# for gpu in tf.config.experimental.list_physical_devices("GPU"):
#     tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras import layers, Model, optimizers, metrics
import tensorflow.keras.backend as K
import numpy as np
import time
import log


from train_log_formatter import print_status_bar


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def accuracy(y_true, y_pred, threshold):  # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))


def calculate_tp_fp(y_true, y_pred, threshold):
    # y_pred等同于dist
    predict_is_same = K.cast(y_pred < threshold, y_true.dtype)
    true_accept = np.sum(np.logical_and(predict_is_same, y_true))
    false_accept = np.sum(np.logical_and(predict_is_same, np.logical_not(y_true)))
    return true_accept, false_accept


class TripLossModel:
    def __init__(self, backbone, input_shape, nof_class_per_batch, nof_phases_per_class, margin):
        self.backbone: Model = backbone
        self.input_shape = input_shape
        self.nof_class_per_batch, self.nof_phases_per_class = nof_class_per_batch, nof_phases_per_class
        self.margin = margin

        self.optimizer = optimizers.Adam()

        self.l2_norm = layers.Lambda(lambda embeddings: K.l2_normalize(embeddings, axis=1), name='l2_norm')
        self.model = self._construct_model(self.input_shape)

    def _construct_model(self, input_shape):
        input = layers.Input(shape=input_shape)
        # size = class_per_batch * phases_per_class
        embeddings = self.backbone(input)
        embeddings = self.l2_norm(embeddings)
        model = Model(inputs=input, outputs=embeddings)
        return model

    def triplet_loss(self, y_pred):
        assert len(y_pred) % 3 == 0
        anchor = y_pred[::3]
        pos = y_pred[1::3]
        neg = y_pred[2::3]
        pos_dist = K.sum(K.square(anchor - pos), axis=1)
        neg_dist = K.sum(K.square(anchor - neg), axis=1)
        basic_loss = pos_dist - neg_dist + self.margin
        loss = K.mean(K.maximum(basic_loss, 0))

        return loss

    def train(self, data_loader, val_dataset=None, epochs=1000, steps=100):
        mean_train_loss = metrics.Mean()
        for epoch in range(epochs):
            # reset states
            mean_train_loss.reset_states()

            print("Epoch {}/{}".format(epoch + 1, epochs))
            start_time = time.time()
            for step in range(steps):
                # batch_size = class_per_batch * phases_per_class
                batch_dataset = data_loader.get_random_batch(self.nof_class_per_batch, self.nof_phases_per_class)
                # select triplets
                # training=True和trainging=False的结果不同，和dropout以及BN等有关
                embeddings_before_training = self.model(batch_dataset, training=False)
                # l2，移动到了model中去
                # embeddings_before_training = K.l2_normalize(embeddings_before_training, axis=1)
                triplets_idx = self._select_triplets_idx(embeddings_before_training, self.margin)
                if len(triplets_idx) == 0:
                    log.logger.warn('no triplets')
                    continue
                input_triplet = batch_dataset[triplets_idx]
                # with tf.GradientTape() as tape:
                #     triplet_embeddings = self.model(input_triplet, training=True)
                #     # l2
                #     # triplet_embeddings = K.l2_normalize(triplet_embeddings, axis=1)
                #     triplet_loss = self.triplet_loss(triplet_embeddings)
                #     # self.model.losses是一个数组，可能是一个空数组，和l2正则化之类的有关
                #     loss = tf.add_n([triplet_loss] + self.model.losses)
                # gradients = tape.gradient(loss, self.model.trainable_variables)
                # self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                # mean_train_loss(loss)
                # 再分batch，可以移到config
                batch_size = 32 * 3
                for i in range(input_triplet.shape[0]//batch_size+1):
                    batch_input = input_triplet[i*batch_size:(i+1)*batch_size]
                    if len(batch_input) == 0:
                        log.logger.warn('no batch')
                        continue
                    with tf.GradientTape() as tape:
                        triplet_embeddings = self.model(batch_input, training=True)
                        # l2
                        # triplet_embeddings = K.l2_normalize(triplet_embeddings, axis=1)
                        triplet_loss = self.triplet_loss(triplet_embeddings)
                        # self.model.losses是一个数组，可能是一个空数组，和l2正则化之类的有关
                        loss = tf.add_n([triplet_loss] + self.model.losses)
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    mean_train_loss(loss)
                # evaluate
            if val_dataset is not None:
                thresholds = np.arange(0, 4, 0.01)
                nof_thresholds = len(thresholds)
                acc_test = np.zeros(nof_thresholds)
                tp_test = np.zeros(nof_thresholds)
                fp_test = np.zeros(nof_thresholds)
                n_same, n_diff = 0, 0
                total_batch = 0
                for te_X, te_Y in val_dataset:
                    embeddings1 = self.model(te_X[:, 0], training=False)
                    embeddings2 = self.model(te_X[:, 1], training=False)
                    dist = euclidean_distance([embeddings1, embeddings2])
                    n_same += np.sum(te_Y)
                    n_diff += np.sum(np.logical_not(te_Y))
                    for threshold_idx, threshold in enumerate(thresholds):
                        acc_test[threshold_idx] += accuracy(te_Y, dist, threshold)
                        tp, fp = calculate_tp_fp(te_Y, dist, threshold)
                        tp_test[threshold_idx] += tp
                        fp_test[threshold_idx] += fp
                    total_batch += 1
                acc_test /= total_batch
                # 获得最好threshold
                best_threshold_index = np.argmax(acc_test)
                best_threshold = thresholds[best_threshold_index]
                best_acc = acc_test[best_threshold_index]
                tp_test /= n_same
                fp_test /= n_diff
                best_val = tp_test[best_threshold_index]
                best_far = fp_test[best_threshold_index]
                # 计算auc
                auc = np.sum(((tp_test[:-1] + tp_test[1:]) * np.diff(fp_test)) / 2)
                # 计算平均acc
                mean_acc = (tp_test + 1 - fp_test) / 2
                best_mean_acc = np.max(mean_acc)

                print(f'- best_threshold: {best_threshold} - best_acc: {best_acc:.4f}'
                      f' - best_val: {best_val:.4f} - best_far: {best_far:.4f} - auc: {auc:.4f} - best_mean_acc: {best_mean_acc:.4f}')

            end_time = time.time()
            print_status_bar(end_time - start_time, mean_train_loss)

    def _select_triplets_idx(self, embeddings, alpha):
        emb_start_idx = 0
        triplets = []
        for i in range(self.nof_class_per_batch):
            for j in range(self.nof_phases_per_class):
                anchor_idx = emb_start_idx + j
                # 得到anchor和batch中其他所有的l2距离
                all_dists_sqrs = np.sum(np.square(embeddings[anchor_idx].numpy() - embeddings.numpy()), 1)
                # 遍历每个positive
                for k in range(j + 1, self.nof_phases_per_class):
                    positive_idx = emb_start_idx + k
                    pos_dist_sqr = np.sum(np.square(embeddings[anchor_idx] - embeddings[positive_idx]))
                    all_dists_sqrs[emb_start_idx:emb_start_idx + self.nof_phases_per_class] = np.NaN
                    all_neg_idxs = np.where(all_dists_sqrs - pos_dist_sqr < alpha)[0]

                    nof_random_negs = all_neg_idxs.shape[0]
                    if nof_random_negs > 0:
                        rnd_idx = np.random.randint(nof_random_negs)
                        negative_idx = all_neg_idxs[rnd_idx]
                        triplets.append([anchor_idx, positive_idx, negative_idx])

            emb_start_idx += self.nof_phases_per_class
        np.random.shuffle(triplets)
        return np.reshape(triplets, -1)

    def save_weights(self, weights_path):
        self.model.save_weights(weights_path)
