import pandas as pd
import tensorflow as tf

print(tf.__version__)

# 将 iris.csv 保存成TFRecord文件
input_csv_file = "iris.csv"
iris_frame = pd.read_csv(input_csv_file, header=0)
print(iris_frame)
# label,sepal_length,sepal_width,petal_length,petal_width
print("values shape: ", iris_frame.shape)

row_count = iris_frame.shape[0]
col_count = iris_frame.shape[1]

output_tfrecord_file = "iris.tfrecords"
with  tf.io.TFRecordWriter(output_tfrecord_file) as writer:
    for i in range(row_count):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[iris_frame.iloc[i, 0]])),
                    "sepal_length": tf.train.Feature(float_list=tf.train.FloatList(value=[iris_frame.iloc[i, 1]])),
                    "sepal_width": tf.train.Feature(float_list=tf.train.FloatList(value=[iris_frame.iloc[i, 2]])),
                    "petal_length": tf.train.Feature(float_list=tf.train.FloatList(value=[iris_frame.iloc[i, 3]])),
                    "petal_width": tf.train.Feature(float_list=tf.train.FloatList(value=[iris_frame.iloc[i, 4]]))

                }
            )
        )
        writer.write(record=example.SerializeToString())
    writer.close()