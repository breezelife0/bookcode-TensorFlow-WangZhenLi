import tensorflow as tf
# 转换模型
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()
# 保存模型
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
