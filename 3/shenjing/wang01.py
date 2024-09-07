import tensorflow as tf
from matplotlib import pyplot as plt
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False
# 创建 W,b 张量
x = tf.random.normal([2,784])
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
# 线性变换
o1 = tf.matmul(x,w1) + b1
# 激活函数
o1 = tf.nn.relu(o1)

x = tf.random.normal([4,28*28])
# 导入层模块
from tensorflow.keras import layers
# 创建全连接层，指定输出节点数和激活函数
fc = layers.Dense(512, activation=tf.nn.relu)
# 通过 fc 类实例完成一次全连接层的计算，返回输出张量
h1 = fc(x)

# 获取 Dense 类的权值矩阵
print(fc.kernel)
# 获取 Dense 类的偏置向量
print(fc.bias)
# 待优化参数列表
print(fc.trainable_variables)
# 返回所有参数列表
print(fc.variables)

# 隐藏层 1 张量
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
# 隐藏层 2 张量
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
# 隐藏层 3 张量
w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
# 输出层张量
w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))
with tf.GradientTape() as tape: # 梯度记录器
    # x: [b, 28*28]
    # 隐藏层 1 前向计算， [b, 28*28] => [b, 256]
    h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
    h1 = tf.nn.relu(h1)
    # 隐藏层 2 前向计算， [b, 256] => [b, 128]
    h2 = h1@w2 + b2
    h2 = tf.nn.relu(h2)
    # 隐藏层 3 前向计算， [b, 128] => [b, 64]
    h3 = h2@w3 + b3
    h3 = tf.nn.relu(h3)
    # 输出层前向计算， [b, 64] => [b, 10]
    h4 = h3@w4 + b4

# 导入常用网络层 layers
from tensorflow.keras import layers
# 隐藏层 1
fc1 = layers.Dense(256, activation=tf.nn.relu)
# 隐藏层 2
fc2 = layers.Dense(128, activation=tf.nn.relu)
# 隐藏层 3
fc3 = layers.Dense(64, activation=tf.nn.relu)
# 输出层
fc4 = layers.Dense(10, activation=None)

x = tf.random.normal([4,28*28])
# 通过隐藏层 1 得到输出
h1 = fc1(x)
# 通过隐藏层 2 得到输出
h2 = fc2(h1)
# 通过隐藏层 3 得到输出
h3 = fc3(h2)
# 通过输出层得到网络输出
h4 = fc4(h3)

# 导入 Sequential 容器
from tensorflow.keras import layers,Sequential
# 通过 Sequential 容器封装为一个网络类
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu) , # 创建隐藏层 1
    layers.Dense(128, activation=tf.nn.relu) , # 创建隐藏层 2
    layers.Dense(64, activation=tf.nn.relu) , # 创建隐藏层 3
    layers.Dense(10, activation=None) , # 创建输出层
])

out = model(x) # 前向计算得到输出)

# 构造-6~6 的输入向量
x = tf.linspace(-6.,6.,10)
print(x)

# 通过 Sigmoid 函数
sigmoid_y = tf.nn.sigmoid(x)
print(sigmoid_y)

def set_plt_ax():
    # get current axis 获得坐标轴对象
    ax = plt.gca()

    ax.spines['right'].set_color('none')
    # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    # 指定下边的边作为 x 轴，指定左边的边为 y 轴
    ax.yaxis.set_ticks_position('left')

    # 指定 data  设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

set_plt_ax()
plt.plot(x, sigmoid_y, color='C4', label='Sigmoid')
plt.xlim(-6, 6)
plt.ylim(0, 1)
plt.legend(loc=2)
plt.show()



