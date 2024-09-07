import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # 手写数字识别数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 数据归一化处理
x_train, x_test = x_train / 255.0, x_test / 255.0
imgs = x_test[:3] # 查看测试集前3张图像
labs = y_test[:3] # 查看标签
print(labs)
# np.hstack():将图像在水平方向上平铺
plot_imgs = np.hstack(imgs)
plt.imshow(plot_imgs,cmap = 'gray')
plt.show()