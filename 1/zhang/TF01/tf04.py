import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, metrics

path = 'MNIST/mnist.npz'
with np.load(path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print(f'dataset info: shape: {x_train.shape}, {y_train.shape}')
print(f'dataset info: max: {x_train.max()}')
print(f'dataset info: min: {x_train.min()}')

print("A sample:")
print("y_train: ", y_train[0])
# print("x_train: \n", x_train[0])
show_pic = x_train[0].copy()
show_pic = cv2.resize(show_pic, (28 * 10, 28 * 10))
cv2.imshow("A image sample", show_pic)
key = cv2.waitKey(0)
# 按 q 退出
if key == ord('q'):
    cv2.destroyAllWindows()
    print("show demo over")