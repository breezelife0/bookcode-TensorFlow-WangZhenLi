import numpy as np
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28), name='input'),
    tf.keras.layers.LSTM(20, time_major=False, return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='output')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# 如果要快速测试流，请将其更改为True。
# #使用小数据集和仅1个epoch进行训练。该模型将工作得很差，但这提供了一种测试转换是否端到端工作的快速方法。
_FAST_TRAINING = False
_EPOCHS = 5
if _FAST_TRAINING:
  _EPOCHS = 1
  _TRAINING_DATA_COUNT = 1000
  x_train = x_train[:_TRAINING_DATA_COUNT]
  y_train = y_train[:_TRAINING_DATA_COUNT]

model.fit(x_train, y_train, epochs=_EPOCHS)
model.evaluate(x_test, y_test, verbose=0)

run_model = tf.function(lambda x: model(x))
# 这很重要，让我们修正输入大小。
BATCH_SIZE = 1
STEPS = 28
INPUT_SIZE = 28
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model.inputs[0].dtype))

#保存模型的目录
MODEL_DIR = "keras_lstm"
model.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
tflite_model = converter.convert()

#使用TensorFlow运行模型以获得预期结果.
TEST_CASES = 10

#使用TensorFlow Lite运行模型
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for i in range(TEST_CASES):
  expected = model.predict(x_test[i:i+1])
  interpreter.set_tensor(input_details[0]["index"], x_test[i:i+1, :, :])
  interpreter.invoke()
  result = interpreter.get_tensor(output_details[0]["index"])

  #断言TFLite模型的结果是否与TF模型一致。
  np.testing.assert_almost_equal(expected, result)
  print("Done. The result of TensorFlow matches the result of TensorFlow Lite.")

  # 请注意：TfLite融合的Lstm内核是有状态的，因此我们需要重置状态。
  # 清理内部状态
  interpreter.reset_all_variables()