import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import dataset_ops as dops
from dataset_ops import to_onehot
import math


# read data
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = dops.load_data_jay_lyrics()

# parameter initialize
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size


def get_params():
    # model parameters' setting
    def _one(shape):
        return tf.Variable(tf.random.normal(shape=shape, stddev=0.01, dtype=tf.float32))

    # hidden parameter
    w_xh = _one((num_inputs, num_hiddens))
    w_hh = _one((num_hiddens, num_hiddens))
    b_h = tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32)

    # output parameter
    w_hq = _one((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)

    params = [w_xh, w_hh, b_h, w_hq, b_q]
    return params


def init_rnn_state(batch_size, num_hiddens):
    # initial hidden status
    return (tf.zeros(shape=(batch_size, num_hiddens)),)


def rnn(inputs, state, params):
    # shape (batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state

    outputs = []
    for x in inputs:
        x = tf.reshape(x, shape=[-1, W_xh.shape[0]])
        # rnn ops
        h = tf.tanh(tf.matmul(x, W_xh) + tf.matmul(H, W_hh) + b_h)

        # basement ops
        y = tf.matmul(h, W_hq) + b_q

        outputs.append(y)

    return outputs, (H,)


# predict the next string by prefix
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = tf.convert_to_tensor(to_onehot(np.array([output[-1]]), vocab_size), dtype=tf.float32)
        X = tf.reshape(X, [1, -1])
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(np.array(tf.argmax(Y[0], axis=1))))
    # print(output)
    # print([idx_to_char[i] for i in output])
    return ''.join([idx_to_char[i] for i in output])


def grad_clipping(grads, theta):
    # calculate the l2 norm of current gradients
    norm = np.array([0])
    for i in range(len(grads)):
        norm += tf.reduce_sum(grads[i]**2)

    norm = np.sqrt(norm).item()
    new_gradient = []
    if norm > theta:
        for grad in grads:
            new_gradient.append(grad * theta / norm)
    else:
        for grad in grads:
            new_gradient.append(grad)

    return new_gradient


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size,  corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = dops.data_iter_random
    else:
        data_iter_fn = dops.data_iter_consecutive

    params = get_params()
    optimizer = tf.keras.optimizers.SGD(lr=lr)

    for epoch in range(num_epochs):
        # initial hidden status before epoch if use the consecutive iter
        if not is_random_iter:
            state = init_rnn_state(batch_size, num_hiddens)

        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps)

        for X,Y in data_iter:
            # initial hidden status before mini_batch if use the random iter
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(params)
                inputs = to_onehot(X, vocab_size)

                # num_steps * (batch_size, vocab_size)
                (outputs, state) = rnn(inputs, state, params)
                outputs = tf.concat(outputs, 0)

                # transform to the same structure
                y = Y.T.reshape((-1, ))
                y = tf.convert_to_tensor(y, dtype=tf.float32)

                l = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y, outputs))

            grads = tape.gradient(l, params)
            grads = grad_clipping(grads, clipping_theta)
            optimizer.apply_gradients(zip(grads, params))

            l_sum += np.array(l).item() * len(y)
            n += len(y)

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            # print(params)
            for prefix in prefixes:
                print(prefix)
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, idx_to_char, char_to_idx))


if __name__ == '__main__':
    # X = np.arange(10).reshape((2, 5))
    # state = init_rnn_state(X.shape[0], num_hiddens)
    # inputs = to_onehot(X, vocab_size)
    # params = get_params()
    # outputs, state_new = rnn(inputs, state, params)
    # print(len(outputs), outputs[0].shape, state_new[0].shape)
    #
    # print(predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
    #                   idx_to_char, char_to_idx))

    num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

    # train with random sampling
    # train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
    #                       vocab_size, corpus_indices, idx_to_char,
    #                       char_to_idx, True, num_epochs, num_steps, lr,
    #                       clipping_theta, batch_size, pred_period, pred_len,
    #                       prefixes)

    # train with consecutive sampling
    train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)


