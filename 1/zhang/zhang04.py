import tensorflow as tf

#三维张量，3行4列深度为2的张量
rank_3_tensor= tf.constant([
                     [[ 1,  2], [ 3,  4], [ 5,  6], [ 7,  8]],
                     [[11, 12], [13, 14], [15, 16], [17, 18]],
                     [[21, 22], [23, 24], [25, 26], [27, 28]]
                     ], tf.float16)
print(rank_3_tensor)