import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#预处理数据（这些都是NumPy数组）
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

#保留10000个样品进行验证
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model

model = get_compiled_model()

# 首先，让我们创建一个训练数据集实例，接下来将使用与前面相同的MNIST数据。
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# 洗牌并切片数据集.
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

# 现在我们得到了一个测试数据集.
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64)

#由于数据集已经处理批处理，所以我们不传递“batch\u size”参数。
model.fit(train_dataset, epochs=3)

#还可以对数据集进行评估或预测.
print("Evaluate评估:")
result = model.evaluate(test_dataset)
dict(zip(model.metrics_names, result))