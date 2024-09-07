import tensorflow as tf
import numpy as np

# x*w_xh+h*w_hh == [x,h, axis=column]*[w_xh, w_hh, axis=row]
x, w_xh = tf.random.normal(shape=[3, 1]), tf.random.normal(shape=[1, 4])
h, w_hh = tf.random.normal(shape=[3, 4]), tf.random.normal(shape=[4, 4])

ops1 = tf.matmul(x, w_xh) + tf.matmul(h, w_hh)
ops2 = tf.matmul(tf.concat([x, h], axis=1), tf.concat([w_xh, w_hh], axis=0))

# equal equation
print(ops1 == ops2)
print(tf.equal(ops1, ops2))
