import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设

print(tf.__version__)
print(np.__version__)

def generate_data(epochs, batch_size=100):
    """y = 2x 函数数据生成器 增加迭代器"""
    for i in range(epochs):
        x_batch = np.linspace(-1, 1, batch_size)    # 为-1到1之间连续的100个浮点数
    #     print("*x_batch.shape", *x_batch.shape)
        y_batch = 2 * x_batch + np.random.randn(x_batch.shape[0]) * 0.3     # y=2x，但是加入了噪声

        yield shuffle(x_batch, y_batch), i        # 以生成器的方式返回

# 1.循环获取数据
train_epochs = 10

for (x_batch, y_batch), epoch_index in generate_data(train_epochs):
    x_batch = tf.cast(x_batch, tf.float32)
    y_batch = tf.cast(y_batch, tf.float32)
    print(epoch_index, "| x.shape:", x_batch.shape, "| x[:3]:", x_batch[:3].numpy())
    print(epoch_index, "| y.shape:", y_batch.shape, "| y[:3]:", y_batch[:3].numpy())

# 2.显示一组数据
train_data = list(generate_data(1))[0]
plt.plot(train_data[0][0], train_data[0][1], 'ro', label='Original data')
plt.legend()
plt.show()
