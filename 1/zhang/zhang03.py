import tensorflow as tf

#二维张量，3行4列
rank_2_tensor  = tf.constant([
                     [1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]
                     ], tf.float16)
print(rank_2_tensor)