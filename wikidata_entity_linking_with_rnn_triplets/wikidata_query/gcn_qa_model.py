import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

TINY = 1e-6
ONE = tf.constant(1.)
NAMESPACE = 'gcn_qa'
forbidden_weight = 1.
_weight_for_positive_matches = 1.
_rw = 1e-1


class GCN_QA(object):
    _nodes_vocab_size = 300 * 3
    _question_vocab_size = 300
    _question_vector_size = 150
    _types_size = 3
    _mask_size = 200
    _types_proj_size = 5
    _word_proj_size = 50
    _word_proj_size_for_rnn = 50
    _word_proj_size_for_item = 50
    _internal_proj_size = 250
    _hidden_layer1_size = 250
    _hidden_layer2_size = 250
    _output_size = 2

    _memory_dim = 100
    _stack_dimension = 2

    def __init__(self, dropout=1.0):
        tf.reset_default_graph()
        with tf.variable_scope(NAMESPACE):
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=config)

            # Input variables
            self.node_X_fw = tf.placeholder(tf.float32, shape=(None, None, self._nodes_vocab_size), name='node_X_fw')
            self.node_X_bw = tf.placeholder(tf.float32, shape=(None, None, self._nodes_vocab_size), name='node_X_bw')
            self.question_vectors_fw = tf.placeholder(tf.float32, shape=(None, None, self._question_vocab_size),
                                                      name='question_vectors_inp_fw')
            self.question_vectors_bw = tf.placeholder(tf.float32, shape=(None, None, self._question_vocab_size),
                                                      name='question_vectors_inp_nw')
            self.question_mask = tf.placeholder(tf.float32, shape=(None, None, self._mask_size),
                                                name='question_mask')

            # The question is pre-processed by a bi-GRU
            self.Wq = tf.Variable(tf.random_uniform([self._question_vocab_size,
                                                     self._word_proj_size_for_rnn], -_rw, _rw))
            self.bq = tf.Variable(tf.random_uniform([self._word_proj_size_for_rnn], -_rw, _rw))
            self.internal_projection = lambda x: tf.nn.relu(tf.matmul(x, self.Wq) + self.bq)
            self.question_int_fw = tf.map_fn(self.internal_projection, self.question_vectors_fw)
            self.question_int_bw = tf.map_fn(self.internal_projection, self.question_vectors_bw)

            self.rnn_cell_fw = rnn.MultiRNNCell([rnn.GRUCell(self._memory_dim) for _ in range(self._stack_dimension)],
                                                state_is_tuple=True)
            self.rnn_cell_bw = rnn.MultiRNNCell([rnn.GRUCell(self._memory_dim) for _ in range(self._stack_dimension)],
                                                state_is_tuple=True)
            with tf.variable_scope('fw'):
                output_fw, state_fw = tf.nn.dynamic_rnn(self.rnn_cell_fw, self.question_int_fw, time_major=True,
                                                        dtype=tf.float32)
            with tf.variable_scope('bw'):
                output_bw, state_bw = tf.nn.dynamic_rnn(self.rnn_cell_bw, self.question_int_bw, time_major=True,
                                                        dtype=tf.float32)

            self.states = tf.concat(values=[output_fw, tf.reverse(output_bw, [0])], axis=2)
            self.question_vector_pre = tf.reduce_mean(tf.multiply(self.question_mask, self.states), axis=0)
            self.Wqa = tf.Variable(
                tf.random_uniform([2 * self._memory_dim, self._question_vector_size], -_rw, _rw),
                name='Wqa')
            self.bqa = tf.Variable(tf.random_uniform([self._question_vector_size], -_rw, _rw), name='bqa')
            self.question_vector = tf.nn.relu(tf.matmul(self.question_vector_pre, self.Wqa) + self.bqa)

            # bi-LSTM of triplets part
            self.Wtri = tf.Variable(tf.random_uniform([self._nodes_vocab_size,
                                                       self._word_proj_size_for_rnn], -_rw, _rw))
            self.btri = tf.Variable(tf.random_uniform([self._word_proj_size_for_rnn], -_rw, _rw))
            self.internal_projection = lambda x: tf.nn.relu(tf.matmul(x, self.Wtri) + self.btri)
            self.node_int_fw = tf.map_fn(self.internal_projection, self.node_X_fw)
            self.node_int_bw = tf.map_fn(self.internal_projection, self.node_X_bw)

            self.rnn_cell_fw_tri = rnn.MultiRNNCell(
                [rnn.GRUCell(self._memory_dim) for _ in range(self._stack_dimension)],
                state_is_tuple=True)
            self.rnn_cell_bw_tri = rnn.MultiRNNCell(
                [rnn.GRUCell(self._memory_dim) for _ in range(self._stack_dimension)],
                state_is_tuple=True)
            with tf.variable_scope('fw_tri'):
                output_fw_tri, _ = tf.nn.dynamic_rnn(self.rnn_cell_fw_tri, self.node_int_fw, time_major=True,
                                                     dtype=tf.float32)
            with tf.variable_scope('bw_tri'):
                output_bw_tri, _ = tf.nn.dynamic_rnn(self.rnn_cell_bw_tri, self.node_int_bw, time_major=True,
                                                     dtype=tf.float32)

            self.states_tri = tf.concat(values=[output_fw_tri[-1], output_bw_tri[-1]], axis=1)
            self.Wtn = tf.Variable(
                tf.random_uniform([2 * self._memory_dim, self._question_vector_size], -_rw, _rw),
                name='Wtn')
            self.btn = tf.Variable(tf.random_uniform([self._question_vector_size], -_rw, _rw), name='btn')
            self.first_node = tf.nn.relu(tf.matmul(self.states_tri, self.Wtn) + self.btn)

            self.concatenated = tf.concat(values=[self.question_vector, self.first_node], axis=1)

            # Final feedforward layers
            self.Ws1 = tf.Variable(
                tf.random_uniform([self._question_vector_size
                                   + self._question_vector_size,
                                   self._hidden_layer2_size], -_rw, _rw),
                name='Ws1')
            self.bs1 = tf.Variable(tf.random_uniform([self._hidden_layer2_size], -_rw, _rw), name='bs1')
            self.first_hidden = tf.nn.relu(tf.matmul(self.concatenated, self.Ws1) + self.bs1)
            self.first_hidden_dropout = tf.nn.dropout(self.first_hidden, dropout)

            self.Wf = tf.Variable(
                tf.random_uniform([self._hidden_layer2_size, self._output_size], -_rw,
                                  _rw),
                name='Wf')
            self.bf = tf.Variable(tf.random_uniform([self._output_size], -_rw, _rw), name='bf')
            self.outputs = tf.nn.softmax(tf.matmul(self.first_hidden_dropout, self.Wf) + self.bf)

            # Loss function and training
            self.y_ = tf.placeholder(tf.float32, shape=(None, self._output_size), name='y_')
            self.outputs2 = tf.squeeze(self.outputs)
            self.y2_ = tf.squeeze(self.y_)
            self.one = tf.ones_like(self.outputs)
            self.tiny = self.one * TINY
            self.cross_entropy = (tf.reduce_mean(
                -tf.reduce_sum(self.y_ * tf.log(self.outputs + self.tiny) * _weight_for_positive_matches
                               + (self.one - self.y_) * tf.log(
                    self.one - self.outputs + self.tiny))
            ))

        # Clipping the gradient
        optimizer = tf.train.AdamOptimizer(1e-4)
        gvs = optimizer.compute_gradients(self.cross_entropy)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if var.name.find(NAMESPACE) != -1]
        self.train_step = optimizer.apply_gradients(capped_gvs)
        self.sess.run(tf.global_variables_initializer())

        # Adding the summaries
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./train', self.sess.graph)

    def __train(self, node_X, item_vector, question_vectors, question_mask, y):
        node_X_fw = np.array(node_X)
        node_X_fw = np.transpose(node_X_fw, (1, 0, 2))
        node_X_bw = node_X_fw[::-1, :, :]

        question_vectors = np.array(question_vectors)
        question_vectors_fw = np.transpose(question_vectors, (1, 0, 2))
        question_vectors_bw = question_vectors_fw[::-1, :, :]

        question_mask = np.array(question_mask)
        question_mask = np.transpose(question_mask, (1, 0, 2))

        y = np.array(y)

        feed_dict = {}
        feed_dict.update({self.node_X_fw: node_X_fw})
        feed_dict.update({self.node_X_bw: node_X_bw})
        feed_dict.update({self.question_vectors_fw: question_vectors_fw})
        feed_dict.update({self.question_vectors_bw: question_vectors_bw})
        feed_dict.update({self.question_mask: question_mask})
        feed_dict.update({self.y_: y})

        loss, _, summary, outputs2, y2 = self.sess.run(
            [self.cross_entropy, self.train_step, self.merged, self.outputs2, self.y2_], feed_dict)
        return loss, summary

    def train(self, data, epochs=20):
        for epoch in range(epochs):
            loss, _ = self.__train([data[i][0] for i in range(len(data))],
                                   [data[i][1] for i in range(len(data))],
                                   [data[i][2] for i in range(len(data))],
                                   [data[i][3] for i in range(len(data))],
                                   [data[i][4] for i in range(len(data))])
            print(loss)
            sys.stdout.flush()

    def __predict(self, node_X, item_vector, question_vectors, question_mask):
        node_X_fw = np.array(node_X)
        node_X_fw = np.transpose(node_X_fw, (1, 0, 2))
        node_X_bw = node_X_fw[::-1, :, :]

        question_vectors = np.array(question_vectors)
        question_vectors_fw = np.transpose(question_vectors, (1, 0, 2))
        question_vectors_bw = question_vectors_fw[::-1, :, :]

        question_mask = np.array(question_mask)
        question_mask = np.transpose(question_mask, (1, 0, 2))

        feed_dict = {}
        feed_dict.update({self.node_X_fw: node_X_fw})
        feed_dict.update({self.node_X_bw: node_X_bw})

        feed_dict.update({self.question_vectors_fw: question_vectors_fw})
        feed_dict.update({self.question_vectors_bw: question_vectors_bw})
        feed_dict.update({self.question_mask: question_mask})

        y_batch = self.sess.run([self.outputs2], feed_dict)
        return y_batch

    def __standardize_item(self, item):
        if item[0] < item[1]:
            return [0., 1.]
        return [1., 0.]

    def predict(self, node_X, item_vector, question_vectors, question_mask):
        output = self.__predict([node_X], [item_vector], [question_vectors], [question_mask])
        return self.__standardize_item(output[0])

    # Loading and saving functions

    def save(self, filename):
        saver = tf.train.Saver()
        saver.save(self.sess, filename)

    def load_tensorflow(self, filename):
        saver = tf.train.Saver([v for v in tf.global_variables() if NAMESPACE in v.name])
        saver.restore(self.sess, filename)

    @classmethod
    def load(self, filename, dropout=1.0):
        model = GCN_QA(dropout)
        model.load_tensorflow(filename)
        return model
