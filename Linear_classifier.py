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
x_train, x_test = x_train / 255.0, x_test / 255.0
mean_image = np.mean(x_train, axis=0)
x_train = x_train - mean_image
x_test = x_test - mean_image
y_train = tf.keras.utils.to_categorical(y_train, num_classes=K)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=K)
x_train = np.reshape(x_train, (Ntr, Din))
x_test = np.reshape(x_test, (Nte, Din))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
std = 1e-5
w1 = std * np.random.randn(Din, K)
b1 = np.zeros(K)

print("w1:", w1.shape)
print("b1:", b1.shape)
batch_size = Ntr
iterations = 300
lr = 1.4e-2
lr_decay = 0.999
reg = 5e-6
t = 2   #increase to reduce training speed
count = 0
start = time.time()
loss_history = []
train_acc_history = []
val_acc_history = []
images=[]
seed = 0
rng = np.random.default_rng(seed=seed)
for t in range(iterations):
    time.sleep(t/1000)
    count += 1

    indices = np.arange(Ntr)
    rng.shuffle(indices)
    x = x_train[indices]
    y = y_train[indices]
    y_pred = x.dot(w1) + b1
    loss = 1. / batch_size * np.square(y_pred - y).sum() + reg * ( np.sum(w1 * w1))
    loss_history.append(loss)
    if t % 10 == 0:
        print('iteration %d / %d: loss %f' % (t, iterations, loss))
        print('Learning rate -' , 60*count/(time.time()-start) , 'epochs per minute')

    dy_pred = 1. / batch_size * 2.0 * (y_pred - y)
    dw1 = x.T.dot(dy_pred) + reg * w1
    db1 = dy_pred.sum(axis=0)
    w1 -= lr * dw1
    b1 -= lr * db1
    lr *= lr_decay

batch_size=y_pred.shape[0]
K=y_pred.shape[1]
y_pred_test=x_test.dot(w1)+b1
batch_size_test=y_pred_test.shape[0]
K_test=y_pred_test.shape[1]
train_acc = 1.0 - (1/(batch_size*K))*(np.abs(np.argmax(y_train, axis=1) - np.argmax(y_pred, axis=1))).sum()
print('train acc =',train_acc)
test_acc = 1.0 - (1/(batch_size_test*K_test))*(np.abs(np.argmax(y_test, axis=1) - np.argmax(y_pred_test, axis=1))).sum()
print('train acc =',test_acc)

x_axis=np.arange(len(loss_history))
plt.plot(x_axis,loss_history)

plt.show()

for i in range(w1.shape[1]):
    temp=np.reshape(w1[:,i]*255,(32,32,3))
    temp=cv.normalize(temp, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    images.append(temp)
fig,ax=plt.subplots(2,5 ,figsize=(30,10))
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(images[i*5+j],vmin=0,vmax=255)
plt.show()


