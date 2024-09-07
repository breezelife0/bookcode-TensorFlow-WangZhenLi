import tensorflow as tf
import IPython.display as display

# 将值转换为与tf.Example兼容的类型
def _bytes_feature(value):
  """  从字符串/字节返回bytes_list"""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList不会从张量中解包字符串.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """从float/double返回一个float_list"""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """从bool/enum/int/uint返回int64_list"""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

cat_in_snow  = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

display.display(display.Image(filename=cat_in_snow))
display.display(display.HTML('Image cc-by: &lt;a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg"&gt;Von.grzanka&lt;/a&gt;'))

display.display(display.Image(filename=williamsburg_bridge))
display.display(display.HTML('&lt;a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg"&gt;From Wikimedia&lt;/a&gt;'))

image_labels = {
    cat_in_snow : 0,
    williamsburg_bridge : 1,
}

#这是一个示例，仅使用cat图像。
image_string = open(cat_in_snow, 'rb').read()

label = image_labels[cat_in_snow]

#创建具有相关功能的词典
def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'height': _int64_feature(image_shape[0]),
      'width': _int64_feature(image_shape[1]),
      'depth': _int64_feature(image_shape[2]),
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, label)).split('\n')[:15]:
  print(line)
print('...')

# 将原始图像文件写入“images.tfrecords”。
# 首先，将这两个图像处理为`tf.Example`消息。
# 然后，写入一个“.tfrecords”文件.
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for filename, label in image_labels.items():
    image_string = open(filename, 'rb').read()
    tf_example = image_example(image_string, label)
    writer.write(tf_example.SerializeToString())

raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

#创建描述功能的词典.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
  #使用上面的字典解析输入tf.Example proto
  return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
parsed_image_dataset

for image_features in parsed_image_dataset:
  image_raw = image_features['image_raw'].numpy()
  display.display(display.Image(data=image_raw))