import tensorflow as tf
from buffer import Buffer
import numpy as np


class PolicyNetworkAgent():

    def __init__(self, input_shape, output_shape, lr, buffer_size, seed):

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.buffer = Buffer(buffer_size, input_shape, output_shape)
        self.lr = lr
        self.sess = None
        self.seed = seed

        # applique tjs le meme random
        tf.set_random_seed(42)
        np.random.seed(42)

    def compile(self):

        # init des variables tf
        self.tf_in, self.tf_actions_saved, self.tf_avantage = self._init_tf_var(self.input_shape)

        # model du reseau : type cnn
        self.action, self.logp_as = self._ccn(self.tf_in, self.tf_actions_saved, self.seed)

        # calcul de la fonction a minimiser
        self.actions_loss = self._loss_fct(self.logp_as, self.tf_avantage)

        # optimizer
        self.train_pi = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.actions_loss)

        # calcul de l'enthropy
        self.action_enthropy = tf.reduce_mean(- self.logp_as)

    def predict_action(self, state):
        act = self.sess.run(self.action, feed_dict={self.tf_in: state})
        return act

    def train(self):

        img_s, a_s, avantage = self.buffer.get()
        l_loss = list()
        l_entr = list()
        # print(a_s)
        # print(avantage)

        for i in range(5):
            _, loss, entr = self.sess.run([self.train_pi, self.actions_loss, self.action_enthropy],
                                          feed_dict={self.tf_in: img_s,
                                                     self.tf_actions_saved: a_s,
                                                     self.tf_avantage: avantage})

            l_loss.append(loss)
            l_entr.append(entr)

        print("Entropy : {}, Loss: {} \n".format(np.mean(l_entr), np.mean(l_loss)))

    def set_sess(self, session):
        self.sess = session

    def _init_tf_var(self, input_shape):

        tf_input = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name='images')
        tf_action_saved = tf.placeholder(tf.int32, shape=(None,), name='actions')
        tf_avantage = tf.placeholder(tf.float32, shape=(None,), name='avantages')

        return tf_input, tf_action_saved, tf_avantage

    def _ccn(self, tf_in, action_saved, seed=None):

        if seed is not None:
            tf.random.set_random_seed(seed)

        # architecture du reseau de neuronnes
        in_ex = tf.expand_dims(tf_in, axis=3)
        conv1 = tf.layers.conv2d(in_ex, 3, 1, activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, 3, 1, activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, 3, 1, activation=tf.nn.relu)

        flattened = tf.layers.flatten(conv3)
        d1 = tf.layers.dense(flattened, units=256, activation=tf.nn.relu)
        d2 = tf.layers.dense(d1, units=128, activation=tf.nn.relu)
        action_logit = tf.layers.dense(d2, units=self.output_shape)


        # log des differentes actions (pour le calcul de pi = log(p(action))*avantage)
        log_all_actions = tf.nn.log_softmax(action_logit)

        # prediction de l'action, calcul des log de l'action predite et sauvegardee (pour l'apprentissage)
        action_predicted = tf.squeeze(tf.random.categorical(action_logit, 1), axis=1)
        # log_action_predicted = tf.reduce_sum(tf.one_hot(action_predicted, depth=output_shape)*log_all_actions)
        log_action_saved = tf.reduce_sum(tf.one_hot(action_saved, depth=self.output_shape) * log_all_actions)

        return action_predicted, log_action_saved

    def _loss_fct(self, log_actions_saved, tf_avantages):
        return - tf.reduce_mean(log_actions_saved * tf_avantages)



