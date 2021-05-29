from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
import cv2
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, Flatten
from google.colab import drive
import matplotlib.pyplot as plt
from numpy import save, load
import numpy as np
from copy import deepcopy
import tensorflow as tf
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
drive.mount('/content/drive')

def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images

def winning_neuron(x, W):
    # Also called as Best Matching Neuron/Best Matching Unit (BMU)
    return np.argmin(np.linalg.norm(x - W, axis=1))

def update_weights(lr, var, x, W, Grid):
    i = winning_neuron(x, W)
    d = np.square(np.linalg.norm(Grid - Grid[i], axis=1))
    # Topological Neighbourhood Function
    h = np.exp(-d/(2 * var * var))
    W = W + lr * h[:, np.newaxis] * (x - W)
    return W


def decay_learning_rate(eta_initial, epoch, time_const):
    return eta_initial * np.exp(-epoch/time_const)


def decay_variance(sigma_initial, epoch, time_const):
    return sigma_initial * np.exp(-epoch/time_const)

def clu_labels(kmeans, labels):
    print('clu start------------------')
    arr, result, = [0,0,0,0,0,0,0,0,0,0], []
    for i in range(len(np.unique(labels))):
        label_arr = np.array(np.where(kmeans.labels_ == i))
        for l in label_arr:
            for x in l:
                arr[labels[x]-1] += 1
        print(arr)
        while arr.index(max(arr)) in result:
            arr[arr.index(max(arr))] = 0
        result.append(arr.index(max(arr)))
        arr = [0,0,0,0,0,0,0,0,0,0]
    print('end---------------------')
    return result


"""
model = InceptionV3(include_top=False, input_shape=(96, 96, 3))
flat1 = Flatten()(model.layers[-1].output)
model = Model(inputs=model.inputs, outputs=flat1)
model.summary()
"""

model = VGG16(include_top=False, input_shape=(96, 96, 3))
flat1 = Flatten()(model.layers[-1].output)
dense1 = Dense(256, activation='relu')(flat1)
dense2 = Dense(10, activation='softmax')(dense1)
model = Model(inputs=model.inputs, outputs=dense2)
model.summary()



input_data = np.array(load('/content/drive/My Drive/train_data.npy'))
test_x = read_all_images('/content/drive/My Drive/test_X.bin')

with open('train_y.bin', 'rb') as f:
  y_train = np.fromfile(f, dtype=np.uint8)
with open('test_y.bin', 'rb') as f:
  y_test = np.fromfile(f, dtype=np.uint8)
for i in range(len(y_train)):
  y_train[i] -= 1
y_train = np.array(np_utils.to_categorical(y_train))
for i in range(len(y_test)):
  y_test[i] -= 1
y_test = np.array(np_utils.to_categorical(y_test))


datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                             horizontal_flip=True, zoom_range = (0.1,0.1))
it_train = datagen.flow(input_data, y_train, batch_size=32)


sgd = SGD(learning_rate=0.01, decay=5e-4, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(input_data, y_train, batch_size=32, epochs=5, validation_split=0.2)

model = Model(inputs=model.inputs, outputs=dense1)
model.summary()

encoded_data = model.predict(input_data)


Raw_Data_Shape = np.array([len(encoded_data), len(encoded_data[0])])
SOM_Network_Shape = np.array([25, 25])
X = np.random.randint(0, 256, (Raw_Data_Shape[0], Raw_Data_Shape[1]))
X_Norm = X/np.linalg.norm(X, axis=1).reshape(Raw_Data_Shape[0], 1)
W_Initial_Guess = np.random.uniform(0, 1, (SOM_Network_Shape[0]*SOM_Network_Shape[1], Raw_Data_Shape[1]))
W_Initial_Guess_Norm = W_Initial_Guess/np.linalg.norm(W_Initial_Guess, axis=1).reshape(SOM_Network_Shape[0]*SOM_Network_Shape[1], 1)
Index = np.mgrid[0:SOM_Network_Shape[0],0:SOM_Network_Shape[1]].reshape(2, SOM_Network_Shape[0]*SOM_Network_Shape[1]).T

Epoch = 0
Max_Epoch = 20000
eta_0 = 0.1
eta_time_const = 1000
sigma_0 = np.max(SOM_Network_Shape) * 0.5
sigma_time_const = 1000/np.log10(sigma_0)

W_new = deepcopy(W_Initial_Guess_Norm)
eta = deepcopy(eta_0)
sigma = deepcopy(sigma_0)
with tf.device('/device:GPU:0'):
  while Epoch <= Max_Epoch:
      print(Epoch)
      i = np.random.randint(0, Raw_Data_Shape[0])
      W_new = update_weights(eta, sigma, encoded_data[i], W_new, Index)
      eta = decay_learning_rate(eta_0, Epoch, eta_time_const)
      sigma = decay_variance(sigma_0, Epoch, sigma_time_const)
      Epoch +=1
print('Optimal Weights Reached!!!')
save('weight.npy', W_new)
W_new = load('weight.npy')

W_final = deepcopy(W_new)
kmean_pre, x, y, = [],[],[]

for i in range(len(encoded_data)):
    w = winning_neuron(encoded_data[i], W_final)
    A = float(w) % float(SOM_Network_Shape[0])
    B = float(w) // float(SOM_Network_Shape[1])
    x.append(A)
    y.append(B)
    kmean_pre.append([A, B])


with open('train_y.bin', 'rb') as f:
  y_train = np.fromfile(f, dtype=np.uint8)
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=10, max_iter=300)
print('done')

kmeans.fit(kmean_pre)

centers = kmeans.cluster_centers_
centers_x, centers_y = [], []
X = kmeans.predict(kmean_pre)

for i in range(len(centers)):
    centers_x.append(centers[i][0])
    centers_y.append(centers[i][1])

plt.scatter(x[:1], y[:1])
plt.scatter(centers_x, centers_y)
for i in range(len(y_train[:1500])):
    plt.text(x[i], y[i], y_train[i], fontsize=8)
plt.savefig('plot.png')


result = []
c = clu_labels(kmeans, y_train)
print(c)
for i in range(len(y_train)):
  result.append(c[X[i]]+1)


total = len(y_train)
count, correct = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0,'10':0}, 0

for a,b in zip(result, y_train):
    if a == b:
        count[str(a)] +=1
        correct += 1
print(count)
print(float(correct) / float(total))