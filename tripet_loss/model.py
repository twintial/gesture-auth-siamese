import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, metrics
import tensorflow.keras.backend as K
import numpy as np
import time

from train_log_formatter import print_status_bar


def sample_phase(dataset, nof_class_per_batch, nof_phases_per_class):
    pass


class TripLossModel:
    def __init__(self, backbone, input_shape, nof_class_per_batch, nof_phases_per_class, margin):
        self.backbone: Model = backbone
        self.input_shape = input_shape
        self.nof_class_per_batch, self.nof_phases_per_class = nof_class_per_batch, nof_phases_per_class
        self.margin = margin

        self.optimizer = optimizers.Adam()
        self.model = self._construct_model(self.input_shape)

    def _construct_model(self, input_shape):
        input = layers.Input(shape=input_shape, batch_size=self.nof_class_per_batch * self.nof_phases_per_class)
        # size = class_per_batch * phases_per_class
        embeddings = self.backbone(input)
        # select triplets
        model = Model(inputs=input, outputs=embeddings)
        return model

    def triplet_loss(self, y_pred):
        y_pred = K.l2_normalize(y_pred, axis=1)
        anchor = y_pred[:, 0]
        pos = y_pred[:, 1]
        neg = y_pred[:, 2]
        pos_dist = K.sum(K.square(anchor - pos), axis=1)
        neg_dist = K.sum(K.square(anchor - neg), axis=1)
        basic_loss = pos_dist - neg_dist + self.margin
        loss = K.mean(K.maximum(basic_loss, 0))
        return loss

    def train(self, train_x, epochs=1000, steps=100):
        mean_loss = metrics.Mean()
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch+1, epochs))
            start_time = time.time()
            for step in range(steps):
                # batch_size = class_per_batch * phases_per_class
                batch_dataset = sample_phase(train_x, self.nof_class_per_batch, self.nof_phases_per_class)
                with tf.GradientTape() as tape:
                    embeddings = self.model(batch_dataset)
                    triplets = self._select_triplets(embeddings, self.margin)
                    triplets_loss = self.triplet_loss(triplets)
                    # 还是用这个？loss = tf.add_n([main_loss] + model.losses)
                    loss = triplets_loss + self.model.loss
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                mean_loss(loss)
            end_time = time.time()
            print_status_bar(end_time-start_time, mean_loss)






    def _select_triplets(self, embeddings, alpha):
        emb_start_idx = 0
        triplets = []
        for i in tf.range(self.nof_class_per_batch):
            for j in tf.range(self.nof_phases_per_class):
                anchor_idx = emb_start_idx + j
                # 得到anchor和batch中其他所有的l2距离
                all_dists_sqrs = tf.reduce_sum(tf.square(embeddings[anchor_idx] - embeddings), 1)
                # 遍历每个positive
                for k in tf.range(j + 1, self.nof_phases_per_class):
                    positive_idx = emb_start_idx + k
                    pos_dist_sqr = tf.reduce_sum(tf.square(embeddings[anchor_idx] - embeddings[positive_idx]))
                    all_dists_sqrs[emb_start_idx:emb_start_idx + self.nof_phases_per_class].assgin(np.NaN)
                    all_neg_idxs = np.where(all_dists_sqrs - pos_dist_sqr < alpha)[0]

                    nof_random_negs = all_neg_idxs.shape[0]
                    if nof_random_negs > 0:
                        rnd_idx = np.random.randint(nof_random_negs)
                        negative_idx = all_neg_idxs[rnd_idx]
                        triplets.append([embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]])

            emb_start_idx += self.nof_phases_per_class
        np.random.shuffle(triplets)
        return tf.constant(triplets)
