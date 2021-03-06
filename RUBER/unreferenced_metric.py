__author__ = 'liming-vie'

import os
import pickle
import random
import logging
import tensorflow as tf

import data_helpers

logger = logging.getLogger(__name__)


class Unreferenced(object):
    """Unreferenced Metric
    Measure the relatedness between the generated reply and its query using
    neural network
    """

    def __init__(self,
                 query_max_len,
                 reply_max_len,
                 query_w2v_file,
                 reply_w2v_file,
                 gru_num_units,
                 mlp_units,
                 init_learning_rate=1e-4,
                 l2_regular=0.1,
                 margin=0.5,
                 train_dir='train_data/'):
        """
        Initialize related variables and construct the neural network graph.

        Args:
            qmax_length, rmax_length: max sequence length for query and reply
            fqembed, frembed: embedding matrix file for query and reply
            gru_num_units: number of units in each GRU cell
            mlp_units: number of units for mlp, a list of length T,
                indicating the output units for each perceptron layer.
                No need to specify the output layer size 1.
        """

        # initialize variables
        self.train_dir = train_dir
        self.query_max_length = query_max_len
        self.reply_max_length = reply_max_len
        random.seed()

        print('Loading embedding matrix')
        query_embedding = pickle.load(open(query_w2v_file, 'rb'))
        reply_embedding = pickle.load(open(reply_w2v_file, 'rb'))

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        # graph
        with self.session.as_default():
            # build bidirectional gru rnn and get final state as embedding
            def get_birnn_embedding(sizes, inputs, embed, scope):
                embedding = tf.Variable(embed, dtype=tf.float32, name="embedding_matrix")
                with tf.variable_scope('forward'):
                    fw_cell = tf.contrib.rnn.GRUCell(gru_num_units)
                with tf.variable_scope('backward'):
                    bw_cell = tf.contrib.rnn.GRUCell(gru_num_units)
                inputs = tf.nn.embedding_lookup(embedding, inputs)
                # outputs, state_fw, state_bw
                _, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(
                    fw_cell, bw_cell,
                    # make inputs as [max_length, batch_size=1, vec_dim]
                    tf.unstack(tf.transpose(inputs, perm=[1, 0, 2])),
                    sequence_length=sizes,
                    dtype=tf.float32,
                    scope=scope)
                # [batch_size, gru_num_units * 2]
                return tf.concat([state_fw, state_bw], 1)

            # query GRU bidirectional RNN
            with tf.variable_scope('query_bidirectional_rnn'):
                self.query_sizes = tf.placeholder(tf.int32,
                                                  # batch_size
                                                  shape=[None], name="query_sizes")
                self.query_inputs = tf.placeholder(tf.int32,
                                                   # [batch_size, sequence_length]
                                                   shape=[None, self.query_max_length],
                                                   name="query_inputs")
                with tf.device('/gpu:1'):
                    query_embedding = get_birnn_embedding(
                        self.query_sizes, self.query_inputs,
                        query_embedding, 'query_rgu_birnn')

            # reply GRU bidirectional RNN
            with tf.variable_scope('reply_bidirectional_rnn'):
                self.reply_sizes = tf.placeholder(tf.int32,
                                                  shape=[None], name="reply_sizes")
                self.reply_inputs = tf.placeholder(tf.int32,
                                                   shape=[None, self.reply_max_length],
                                                   name="reply_inputs")
                with tf.device('/gpu:1'):
                    reply_embedding = get_birnn_embedding(
                        self.reply_sizes, self.reply_inputs, reply_embedding, 'reply_gru_birnn')

            # quadratic feature as qT*M*r
            with tf.variable_scope('quadratic_feature'):
                matrix_size = gru_num_units * 2
                M = tf.get_variable('quadratic_M',
                                    shape=[matrix_size, matrix_size],
                                    initializer=tf.zeros_initializer())
                # [batch_size, matrix_size]
                qTM = tf.tensordot(query_embedding, M, 1)
                quadratic = tf.reduce_sum(qTM * reply_embedding, axis=1, keep_dims=True)

            # multi-layer perceptron
            with tf.variable_scope('multi_layer_perceptron'):
                # input layer
                mlp_input = tf.concat(
                    [query_embedding, reply_embedding, quadratic], 1)
                mlp_input = tf.reshape(mlp_input, [-1, gru_num_units * 4 + 1])
                # hidden layers
                inputs = mlp_input
                for i in range(len(mlp_units)):
                    with tf.variable_scope('mlp_layer_%d' % i):
                        inputs = tf.contrib.layers.legacy_fully_connected(
                            inputs, mlp_units[i],
                            activation_fn=tf.tanh,
                            weight_regularizer=tf.contrib.layers.l2_regularizer(l2_regular))
                self.test = inputs
                # dropout layer
                self.training = tf.placeholder(tf.bool, name='training')
                inputs_dropout = tf.layers.dropout(inputs, training=self.training)
                # output layer
                self.score = tf.contrib.layers.legacy_fully_connected(
                    inputs_dropout, 1, activation_fn=tf.sigmoid,
                    weight_regularizer=tf.contrib.layers.l2_regularizer(l2_regular))
                self.score = tf.reshape(self.score, [-1])  # [batch_size]

            # define training related ops
            with tf.variable_scope('train'):
                # calculate losses
                self.pos_score, self.neg_score = tf.split(self.score, 2)
                losses = margin - self.pos_score + self.neg_score
                # make loss >= 0
                losses = tf.clip_by_value(losses, 0.0, 100.0)
                self.loss = tf.reduce_mean(losses)
                # optimizer
                self.learning_rate = tf.Variable(init_learning_rate,
                                                 trainable=False, name="learning_rate")
                self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * 0.99)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.global_step = tf.Variable(0, trainable=False, name="global_step")
                # training op
                with tf.device('/gpu:1'):
                    self.train_op = optimizer.minimize(self.loss, self.global_step)
                # checkpoint saver
                self.saver = tf.train.Saver(tf.global_variables())
                # write summary
                self.log_writer = tf.summary.FileWriter(os.path.join(train_dir, 'logs/'),
                                                        self.session.graph)
                tf.summary.scalar('learning_rate', self.learning_rate)
                tf.summary.scalar('loss', self.loss)
                self.summary = tf.summary.merge_all()

    def get_batch(self, data, data_size, batch_size, idx=None):
        """
        Get a random batch with size batch_size

        Args:
            data: [[length, [ids]], each with a line of segmented sentence
            data_size: size of data
            batch_size: returned batch size
            idx: [batch_size], randomly get batch if idx None, or get with idx

        Return:
            batched vectors [batch_size, max_length]
            sequence length [batch_size]
            idx [batch_size]
        """
        if not idx:
            idx = [random.randint(0, data_size - 1) for _ in range(batch_size)]
            print(idx)
        ids = [data[i][1] for i in idx]
        lens = [data[i][0] for i in idx]
        return ids, lens, idx

    def make_input_feed(self,
                        query_batch, qsizes,
                        reply_batch, rsizes,
                        neg_batch=None, neg_sizes=None,
                        training=True):
        if neg_batch:
            reply_batch += neg_batch
            rsizes += neg_sizes
            # query is all the same.
            query_batch += query_batch
            qsizes += qsizes
        return {
            self.query_sizes: qsizes,
            self.query_inputs: query_batch,
            self.reply_sizes: rsizes,
            self.reply_inputs: reply_batch,
            self.training: training
        }

    def train_step(self, queries, replies, data_size, batch_size):
        query_batch, query_sizes, idx = self.get_batch(queries, data_size, batch_size)
        reply_batch, reply_sizes, _ = self.get_batch(replies, data_size, batch_size, idx)
        negative_reply_batch, neg_reply_sizes, _ = self.get_batch(replies, data_size, batch_size)

        # compute sample loss and do optimize
        feed_dict = self.make_input_feed(query_batch, query_sizes,
                                         reply_batch, reply_sizes,
                                         negative_reply_batch, neg_reply_sizes)
        output_feed = [self.global_step, self.train_op, self.loss, self.summary]
        step, _, loss, summary = self.session.run(output_feed, feed_dict)

        return step, loss, summary

    def init_model(self):
        """
        Initialize all variables or load model from checkpoint
        """
        ckpt = tf.train.get_checkpoint_state(self.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print(('Restoring model from %s' % ckpt.model_checkpoint_path))
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print('Initializing model variables')
            self.session.run(tf.global_variables_initializer())

    def train(self, query_file, reply_file, batch_size=128, steps_per_checkpoint=100):
        queries = data_helpers.load_data(query_file)
        replies = data_helpers.load_data(reply_file)
        data_size = len(queries)

        with self.session.as_default():
            self.init_model()

            checkpoint_path = os.path.join(self.train_dir, "unref.model")
            loss = 0.0
            prev_losses = [1.0]
            while True:
                step, l, summary = self.train_step(queries, replies, data_size, batch_size)
                loss += l
                self.log_writer.add_summary(summary, step)

                # save checkpoint
                if step % steps_per_checkpoint == 0:
                    loss /= steps_per_checkpoint
                    print('global_step %d' % step)
                    print('loss %f' % loss)
                    print('learning_rate %f' % self.learning_rate.eval())

                    if loss > max(prev_losses):
                        self.session.run(self.learning_rate_decay_op)
                    prev_losses = (prev_losses + [loss])[-5:]
                    loss = 0.0

                    # Save and Summary
                    self.saver.save(self.session, checkpoint_path, global_step=self.global_step)

                    # Debug
                    query_batch, query_sizes, idx = self.get_batch(queries, data_size, 10)
                    reply_batch, reply_sizes, idx = self.get_batch(replies, data_size, 10, idx)

                    # No neg_batch when validate.
                    input_feed = self.make_input_feed(query_batch, query_sizes,
                                                      reply_batch, reply_sizes,
                                                      training=False)
                    score, tests = self.session.run([self.pos_score, self.test], input_feed)
                    print('-------------')
                    for s, t in zip(score[:10], tests[:10]):
                        print(s, t)

    def get_scores(self, query_file, reply_file, query_vocab_file, reply_vocab_file, init=False):
        if not init:
            self.init_model()

        queries = data_helpers.load_file(query_file)
        replies = data_helpers.load_file(reply_file)

        query_vocab = data_helpers.load_vocab(query_vocab_file)
        reply_vocab = data_helpers.load_vocab(reply_vocab_file)

        scores = []
        logger.info('looping over query-reply pairs')
        with self.session.as_default():
            for query, reply in zip(queries, replies):
                q_len, q_ids = data_helpers.transform_to_id(query_vocab, query, self.query_max_length)
                r_len, r_ids = data_helpers.transform_to_id(reply_vocab, reply, self.reply_max_length)
                feed_dict = self.make_input_feed([q_ids], [q_len], [r_ids], [r_len], training=False)
                # When training=False there is no neg_score, so as pos_score.
                score = self.session.run(self.score, feed_dict)
                score = float(score[0])
                scores.append(score)
        return scores
