# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 导入MNIST数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 打印训练数据集的格式（60K图像，每个图像为28x28像素）
train_images.shape
len(train_labels)
train_labels # 每个标签都是0到9之间的整数（根据类名^）

#打印测试数据集格式（10K图像，每个图像28x28像素）
test_images.shape
len(test_labels)

#预处理图像（打印信息）
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#缩放值应在0到1之间
train_images = train_images / 255.0
test_images = test_images / 255.0

#使用标签打印训练集中的前25幅图像
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#设置nn层
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 第一层将2D 28x28阵列转换为1D 784阵列（“取消堆叠”阵列并将其排列为一个阵列）
    tf.keras.layers.Dense(128, activation='relu'),  # 第二层（致密层）有128个节点/神经元，每个节点/神经元都有表示当前图像类别的分数
    tf.keras.layers.Dense(10)                       # 第三层（densed layer）返回长度为10的logits（线性输出）数组
])

#编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#通过将标签拟合到训练图像来训练模型
model.fit(train_images, train_labels, epochs=10)

#在测试数据集上运行模型并评估性能
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 附加softmax层将logits转换为概率，然后使用模型进行预测
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

#打印第一个预测（每个num表示模型对图像对应于类的信心）
predictions[0]
np.argmax(predictions[0]) #最可能的类别
test_labels[0]            #实际类别


# 图形预测
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#验证预测（蓝色=正确，红色=不正确）
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# 绘制第一个X测试图像、其预测标签和真实标签。
# 蓝色显示正确预测，红色显示错误预测.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#从测试数据集中获取图像
img = test_images[1]
print(img.shape)

#将图像添加到其为唯一成员的批处理中
img = (np.expand_dims(img,0))
print(img.shape)

#单幅图像的预测
predictions_single = probability_model.predict(img)
print(predictions_single)
np.argmax(predictions_single[0])
print(test_labels[1])

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
