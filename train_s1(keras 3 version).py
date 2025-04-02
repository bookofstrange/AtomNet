# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 08:36:35 2025

@author: JiWei Yu

Keras 3, mainstream in 2025, is incompatible with tensorflow-gpu. 
"""


# %% Import
import os
from keras import layers
# from sklearn.utils import class_weight
import numpy as np
import random
from tensorflow import keras
from keras.utils import to_categorical
import tensorflow as tf
from keras.layers import Activation,Dense,Conv1D,MaxPool1D,BatchNormalization,AveragePooling1D
#%% prepare data

num_f = 32
def data_batch(start,end,random_sample=False,num=5):
    if random_sample==False:
        data_train,label_train=np.load(f'feature/{start}.npy'),np.load(f'label/{start}.npy')
        for i in range(start+1,end):
            # k=ramdom.randint(0,400)
            feature,label=np.load(f'feature/{i}.npy'),np.load(f'label/{i}.npy')
            data_train=np.concatenate((data_train,feature))
            label_train=np.concatenate((label_train,label))
    else:
        k=random.randint(start,end-1)
        data_train,label_train=np.load(f'feature/{k}.npy'),np.load(f'label/{k}.npy')
        for i in range(num-1):
            k=random.randint(start,end-1)
            feature,label=np.load(f'feature/{k}.npy'),np.load(f'label/{k}.npy')
            data_train=np.concatenate((data_train,feature))
            label_train=np.concatenate((label_train,label))
    # label_train = to_categorical(label_train, num_classes=3)
    return data_train[:,:num_f,:],label_train
data_train,label_train=data_batch(0,8)
data_val,label_val=data_batch(8,9)

# y = np.argmax(label_train,axis=1)
# class_weights = class_weight.compute_class_weight(class_weight="balanced",
#                                                   classes=np.unique(y),
#                                                   y=y)
# class_weights = dict(enumerate(class_weights))
#%% pointnet borrowed from https://keras.io/examples/vision/pointnet/

from tensorflow.keras import layers
def conv_bn(x, filters):
    x = Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = BatchNormalization(momentum=0.0)(x)
    return Activation("relu")(x)


def dense_bn(x, filters):
    x = Dense(filters)(x)
    x = BatchNormalization(momentum=0.0)(x)
    return Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "l2reg": self.l2reg,
            'eys':self.eye
        })
        return config

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 128)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 128)
    x = dense_bn(x, 64)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

inputs = keras.Input(shape=(32, 5))

x = tnet(inputs, 5)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 128)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 64)
x = layers.Dropout(0.3)(x)

y = layers.Dense(1, activation="sigmoid")(x)
#%% 

model=keras.Model(inputs=inputs,outputs=y)
model.summary()
# keras.utils.plot_model(
#     model,
#     to_file="AtomNet_1.png",
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=False,
#     rankdir="LR",
#     expand_nested=False,
#     dpi=500,
#     show_layer_activations=True,
#     show_trainable=False,
# )
# %%
# change model here
model_name = 'AtomNet_part1'
# model_name = f'feature{num_f}'
metrics = ['AUC']
checkpoint_save_path=f'./checkpoint/{model_name}.weights.h5'

if os.path.exists(checkpoint_save_path):
    print('-------load the model-------')
    model.load_weights(checkpoint_save_path)

# set callbacks
callback=keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                save_weights_only=True,
                                                save_best_only=True,
                                              )
model.compile(
    # optimizer='adam',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    # loss='categorical_crossentropy',
    metrics=metrics,
    # metrics=[keras.metrics.F1Score()],
)

ep = 1

history=model.fit(data_train,label_train,epochs=ep,batch_size=256
            ,validation_data=(data_val,label_val),
            callbacks=callback,
            # class_weight=class_weights
            )

# %% use test datasets

model.load_weights(checkpoint_save_path)
for i in range(9, 10):
    # i = 'r1(45min)_' + str(i)
    # i = '501r'
    data_test=np.load(f'feature/{i}.npy')
    p=model.predict((data_test),batch_size=256)
    # model.evaluate()
    np.save(f'prediction/{i}.npy',p)
#%% plot training procedure
name=model_name
import matplotlib.pyplot as plt
h_dict=history.history
loss=h_dict['loss']
np.save(f'information/loss{name}.npy',loss)
ep=[i for i in range(ep)]
val_loss=h_dict['val_loss']
np.save(f'information/val_loss{name}.npy',val_loss)
plt.figure(dpi=200)
plt.plot(ep,loss,'go',label='Training loss')
plt.plot(ep,val_loss,'g--',label='Validation_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'information/loss{name}.png')
plt.figure(dpi=200)
acc = h_dict["AUC"]
np.save(f'information/AUC{name}.npy',acc)
val_acc = h_dict["val_AUC"]
np.save(f'information/val_AUC{name}.npy',val_acc)
plt.plot(ep, acc, "bo", label="Training AUC")
plt.plot(ep, val_acc, "b--", label="Validation AUC")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("AUC")
# plt.title(f'average_test_acc:{average_test_acc}')
plt.legend()
plt.savefig(f'information/acc{name}.png')
# %% F1
# plt.figure(dpi=200)
# acc = h_dict["f1_score"]
# # np.save(f'information/AUC{name}.npy',acc)
# val_acc = h_dict["val_f1_score"]
# # val_acc = h_dict["val_f1_score"]
# plt.plot(ep, acc, "bo", label="Training F1")
# plt.plot(ep, val_acc, "b--", label="Validation F1")
# plt.title("Training and validation F1")
# plt.xlabel("Epochs")
# plt.ylabel("F1")
# # plt.title(f'average_test_acc:{average_test_acc}')
# plt.legend()
# plt.savefig(f'information/metrics{name}.png')
