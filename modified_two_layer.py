H = 200
#std = 1e-6
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print("x_train :", x_train.shape)
K = len(np.unique(y_train))  # Classes
Ntr = x_train.shape[0]
Nte = x_test.shape[0]
Din = 3072  # CIFAR10
# Din = 784 # MINIST
# Normalize pixel values
#x_train, x_test = x_train / 255.0, x_test / 255.0
mean_image = np.mean(x_train, axis=0)
x_train = x_train - mean_image
x_test = x_test - mean_image
y_train = tf.keras.utils.to_categorical(y_train, num_classes=K)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=K)
x_train = np.reshape(x_train, (Ntr, Din))
x_test = np.reshape(x_test, (Nte, Din))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

t1 = 2   #increase to reduce training speed
count = 0
start = time.time()
std = 1e-5
w1 = std * np.random.randn(Din, H)
w2 = std * np.random.randn(H,K)
b1 = np.zeros(H)
b2 = np.zeros(K)
print("w1:", w1.shape)
print("b1:", b1.shape)
batch_size = Ntr
iterations = 300
lr = 1.4e-2
lr_decay = 0.999
reg = 5e-6
loss_history = []
train_acc_history = []
val_acc_history = []
seed = 0
rng = np.random.default_rng(seed=seed)
for t in range(iterations):


    indices = np.arange(Ntr)
    rng.shuffle(indices)
    for num in range(100):
        time.sleep(t1 / 1000)
        count += 1
        indices1 = indices[(num*500):((num+1)*500)]
        x = x_train[indices1]
        y = y_train[indices1]
        h = 1.0 / (1.0 + np.exp(-(x.dot(w1) + b1)))
        y_pred = h.dot(w2) + b2
        batch_size1 = x.shape[0]
        loss = 1. / batch_size1 * np.square(y_pred - y).sum() + reg * (np.sum(w2 * w2) + np.sum(w1 * w1))
        loss_history.append(loss)
        if num % 10 == 0:
            print('iteration %d / %d: loss %f' % (t, iterations, loss))
            print('Learning rate -', 60 * count / (time.time() - start), 'epochs per minute')
        dy_pred = 1. / batch_size * 2.0 * (y_pred - y)
        dw2 = h.T.dot(dy_pred) + reg * w2
        db2 = dy_pred.sum(axis=0)
        dh = dy_pred.dot(w2.T)
        dw1 = x.T.dot(dh * h * (1 - h)) + reg * w1
        db1 = (dh * h * (1 - h)).sum(axis=0)
        w1 -= lr * dw1
        w2 -= lr * dw2
        b1 -= lr * db1
        b2 -= lr * db2
        lr *= lr_decay

print('iteration %d / %d: loss %f' % (t, iterations, loss))
print('Learning rate -', 60 * count / (time.time() - start), 'epochs per minute')
batch_size=y_pred.shape[0]
K=y_pred.shape[1]
y_pred_test=x_test.dot(w1)+b1
batch_size_test=y_pred_test.shape[0]
K_test=y_pred_test.shape[1]
train_acc = 1.0 - (1/(batch_size*K))*(np.abs(np.argmax(y_train, axis=1) - np.argmax(y_pred, axis=1))).sum()
print('train acc =',train_acc)
test_acc = 1.0 - (1/(batch_size_test*K_test))*(np.abs(np.argmax(y_test, axis=1) - np.argmax(y_pred_test, axis=1))).sum()
print('test acc =',test_acc)

x_axis=np.arange(len(loss_history))
plt.plot(x_axis,loss_history)

plt.show()