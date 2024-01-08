# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:56:55 2023

@author: JiWei Yu
"""

#%% laod data
import numpy as np
import random
from tensorflow import keras
import tensorflow as tf
from keras.layers import Concatenate,Activation,Dense,Flatten,Input,Dropout,Add,Conv1D,MaxPool1D,BatchNormalization,AveragePooling1D
#%% prepare data
def data_batch(start,end,random_sample=False,num=5):
    if random_sample==False:
        data_train,label_train=np.load(f'feature2/{start}.npy'),np.load(f'label/{start}.npy')
        for i in range(start+1,end):
            # k=ramdom.randint(0,400)
            feature,label=np.load(f'feature2/{i}.npy'),np.load(f'label/{i}.npy')
            data_train=np.concatenate((data_train,feature))
            label_train=np.concatenate((label_train,label))
    else:
        k=random.randint(start,end-1)
        data_train,label_train=np.load(f'feature2/{k}.npy'),np.load(f'label/{k}.npy')
        for i in range(num-1):
            k=random.randint(start,end-1)
            feature,label=np.load(f'feature2/{k}.npy'),np.load(f'label/{k}.npy')
            data_train=np.concatenate((data_train,feature))
            label_train=np.concatenate((label_train,label))
    return data_train[:,:,:-1],label_train


data_train,label_train=data_batch(0,80)
data_val,label_val=data_batch(80,90)
# np.random.seed(0)
# np.random.shuffle(data_train)
# np.random.seed(0)
# np.random.shuffle(label_train)
#%% pointnet
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

inputs = keras.Input(shape=(32, 6))

x = tnet(inputs, 6)
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
#%%
# model_name='PN.keras'
# model=keras.models.load_model(model_name)
#%%
import os
checkpoint_save_path='./checkpoint/AtomNet_2.ckpt'
if os.path.exists(checkpoint_save_path+'.index'):
    print('-------load the model-------')
    model.load_weights(checkpoint_save_path)
callback=keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                save_weights_only=True,
                                                save_best_only=True,
                                              )
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    # metrics=tf.keras.metrics.Recall(
    # thresholds=None, top_k=None, class_id=None, name=None, dtype=None)
    metrics='AUC'
)
ep=10
history=model.fit(data_train,label_train,epochs=ep,batch_size=256
            ,validation_data=(data_val,label_val),
            callbacks=callback
            )

#%%
# model=keras.models.load_model(model_name)
test_acc=[]
for i in range(0,1):
    # i = 'SF' + str(i)
    i = 'SF5'
    data_test=np.load(f'feature2/{i}.npy')[:,:,:-1]
    p=model.predict((data_test),batch_size=512)
    np.save(f'prediction2/{i}.npy',p)

#%%
name='AtomNet_2(append)'
import matplotlib.pyplot as plt
h_dict=history.history
loss=h_dict['loss']
np.save(f'information/loss{name}.npy',loss)
ep=[i for i in range(ep)]
val_loss=h_dict['val_loss']
plt.figure(dpi=200)
plt.plot(ep,loss,'go',label='Training loss')
plt.plot(ep,val_loss,'g--',label='Validation_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'information/loss{name}.png')
plt.figure(dpi=200)
acc = h_dict["auc"]
np.save(f'information/AUC{name}.npy',acc)
val_acc = h_dict["val_auc"]
plt.plot(ep, acc, "bo", label="Training AUC")
plt.plot(ep, val_acc, "b--", label="Validation AUC")
plt.title("Training and validation AUC")
plt.xlabel("Epochs")
plt.ylabel("AUC")
# plt.title(f'average_test_acc:{average_test_acc}')
plt.legend()
plt.savefig(f'information/acc{name}.png')