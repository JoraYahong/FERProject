import os,cv2
import math
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.utils import np_utils
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from cnn_model import simple_CNN
from tensorflow.keras import callbacks
from keras import backend as K
from keras.optimizers import Adam,SGD
import matplotlib.pyplot as plt
import scikitplot
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

data_path = '/mnt/FER/CKplus'
data_dir_list = os.listdir(data_path)
img_rows = 48
img_cols = 48
num_channel = 1
batch_size = 256
epochs = 300

img_data_list = []
for dataset in data_dir_list:
    img_list = os.listdir(data_path + '/' + dataset)
    print('Loaded the images of dataset-' + '{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
        img_data_list.append(input_img)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')

img_data = img_data / 255


num_classes = 7

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')


labels[0:2027]=0 #2028
labels[2028:4822]=1 #2795
labels[4823:8074]=2 #3252
labels[8075:11274]=3 #3200
labels[11275:15483]=4 #4209
labels[15484:17739]=5 #2256
labels[17740:21572]=6 #3833

names = ['disgust','anger','happy','neutral','surprise','fear','sadness']

def getLabel(id):
    return ['disgust','anger','happy','neutral','surprise','fear','sadness'][id]

# shuffle
from sklearn.utils import shuffle
Y = np_utils.to_categorical(labels, num_classes)
x,y = shuffle(img_data,Y, random_state=2)

# split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=2)


# IDG for data augmentation
train_datagen = ImageDataGenerator(
                        featurewise_center=True,
                        featurewise_std_normalization=True,
                        horizontal_flip=True,
                        zca_whitening=True,
                        zca_epsilon=1e-06)


test_datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization = True)
train_generator = train_datagen.flow(X_train,y_train,shuffle = True,seed = 42)
valid_generator = test_datagen.flow(X_test,y_test,shuffle = True,seed = 42)


# for x_batch,y_batch in train_generator:
#     for i in range(8):
#         plt.subplot(2,4,i+1)
#         plt.imshow(x_batch[i], cmap='gray')
#     plt.show()

#clear storage
K.clear_session()

input_shape=(48,48,3)

model = simple_CNN(input_shape,num_classes)
opt=Adam(lr=0.0001,decay=1e-6)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=opt)

filename='trainModel.csv'
filepath="/mnt/FER/model/ckplus.{epoch:02d}.hdf5"



csv_log=callbacks.CSVLogger(filename)
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [csv_log,checkpoint]

# history = model.fit(X_train, y_train,
#                  batch_size=batch_size,
#                  epochs=300,
#                  verbose=1,
#                  validation_data=(X_test, y_test),
#                  callbacks=callbacks_list)

# calculate class weight
def create_class_weight(labels_dict, mu=0.40):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight



# random labels_dict

labels_dict = {0: 2028, 1: 2795, 2: 3252, 3: 3200, 4: 4209, 5: 2256, 6: 3833}
class_weight_create=create_class_weight(labels_dict)
print(class_weight_create)

history = model.fit_generator(train_generator,
        steps_per_epoch=len(X_train)//batch_size,
        epochs=epochs,
        validation_data=valid_generator,
        validation_steps=len(X_test)//batch_size,
        class_weight=class_weight_create,
        callbacks = callbacks_list)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
yhat_valid = model.predict_classes(X_test)
scikitplot.metrics.plot_confusion_matrix(np.argmax(y_test, axis=1), yhat_valid)
plt.savefig("/mnt/FER/confusion_matrix_dcnn.png")
plt.show()
