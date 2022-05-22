# -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""LSTM_Keystroke_Similarity
"""
import numpy as np
import pandas as pd
import random
#import math
import keras.backend as K
#import keras.utils as ku
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import *
from keras.optimizers import Adadelta
from keras.optimizers import Adam
#from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
#from keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.utils import multi_gpu_model

np.random.seed(12345)

# Global Variables
HID_DIM = 128
SQUENCE_NUM = 70
VECTOR_UNIT = 5
VECTOR_LEN = SQUENCE_NUM * VECTOR_UNIT

# Create pairs
def create_pairs(df):
    all_pairs_list = []
    user_list = df['user'].unique().tolist()
    user_num = len(user_list)
    #print('user_num:',user_num)

    for j, u in enumerate(user_list):  # 遍历所有用户
        if j <= 29999:   #second trainning
            continue
        df_user = df[df['user'] == u]
        df_user_len = len(df_user)
        # print('user',u)
        for i in range(df_user_len):
            same_pair_list = []
            diff_pair_list = []

            df_user.sample(frac=1).reset_index(drop=True)  # 打乱用户样本顺序
            z1, z2 = sum(df_user.iloc[i, 2], []), sum(df_user.iloc[(i + 1) % df_user_len, 2], [])  # 同一用户样本
            same_pair_list.append(z1)
            same_pair_list.append(z2)
            same_pair_list.append(1.0)  # 同一用户样本标识1
            all_pairs_list.append(same_pair_list)

            inc = random.randrange(1, user_num)
            dn = (j + inc) % (user_num-1)  # 不同用户样本
            if dn != j:
                diff_user = user_list[dn]
            else:
                diff_user = user_list[(dn + 1) % (user_num-1)]
            # print('diff',diff_user)
            df_diff_user = df[df['user'] == diff_user]
            diff_user_len = len(df_diff_user)
            if diff_user_len == 0:
                '''
                print('dn:',dn)
                print('diff_user:',diff_user)
                print('user_list:',user_list[dn])
                print('user_list_all:',user_list)
                print('user_list_len:', len(user_list))
                print('df_diff_user:',df_diff_user)
                '''
                continue
            z3, z4 = sum(df_user.iloc[(i + 10) % df_user_len, 2], []), sum(df_diff_user.iloc[i % diff_user_len, 2], [])
            diff_pair_list.append(z3)
            diff_pair_list.append(z4)
            diff_pair_list.append(0.0)  # 不同用户样本标识0
            all_pairs_list.append(diff_pair_list)
        '''    
        if j >= 30000:
            break
        '''

    return all_pairs_list


def process_data(pairs):
    keystroke1_data_list = []
    keystroke2_data_list = []
    label_list = []
    for i in range(len(pairs)):
        keystroke1_data = pad_sequences([pairs[i][0]], maxlen=VECTOR_LEN, dtype='float32', padding='post',
                                        truncating='post')
        keystroke2_data = pad_sequences([pairs[i][1]], maxlen=VECTOR_LEN, dtype='float32', padding='post',
                                        truncating='post')
        label = pairs[i][2]
        judge_nan_left = np.array(keystroke1_data[0])
        judge_nan_right = np.array(keystroke2_data[0])
        if np.any(np.isnan(judge_nan_left)) or np.any(np.isnan(judge_nan_right)):
            #print('judge_nan_left',judge_nan_left)
            #print('judge_nan_right',judge_nan_right)
            continue
        keystroke1_data_list.append(keystroke1_data[0])
        keystroke2_data_list.append(keystroke2_data[0])
        label_list.append(label)
        if np.any(np.isnan(judge_nan_left)):
            print('zc_nan')
    return np.array(keystroke1_data_list), np.array(keystroke2_data_list), np.array(label_list)


def Euclidean_distance(vectors):
    left, right = vectors
    #K.print_tensor(left,message='left')
    #K.print_tensor(right,message='right')
    sum_square = K.sum(K.square(left - right), axis=1, keepdims=True)
    #K.print_tensor(sum_square,message='Eucl_dis')
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1.5
    sqaure_pred = K.square(y_pred)
    #margin_square = K.square(K.maximum(K.abs(margin - y_pred), 0.0))
    margin_square = K.square(K.maximum(margin - y_pred, 0.0))

    #K.print_tensor(y_pred,message='y_pred')
    #K.print_tensor(y_true,message='y_true')
    #K.print_tensor(sqaure_pred,message='sqaure_pred')
    #K.print_tensor(margin_square,message='margin_square')
    #K.print_tensor(K.mean(y_true * sqaure_pred + (1.0 - y_true) * margin_square + 1e-8), message='loss result')

    return K.mean(y_true * sqaure_pred + (1.0 - y_true) * margin_square + 1e-10)

def accuracy(y_true, y_pred):  # Tensor上的操作
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 1.0, y_true.dtype)))


def plot_train_history(history, train_metrics, val_metrics):
    plt.plot(history.history.get(train_metrics), '-o')
    plt.plot(history.history.get(val_metrics), '-o')
    plt.ylabel(train_metrics)
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'])


def create_base_network(dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=(SQUENCE_NUM, VECTOR_UNIT))
    mask = Masking(0)(input)
    bn_layer1 = BatchNormalization(name='bn_layer1')(mask)
    lstm_1 = LSTM(dim, activation='relu', kernel_initializer='random_uniform',bias_initializer='zeros',
                  return_sequences=True, name='lstm_layer1', recurrent_dropout=0.3)
    lstm_layer1 = lstm_1(bn_layer1)
    Dropout(lstm_layer1, 0.8)
    bn_layer2 = BatchNormalization(name='bn_layer2_input1')(lstm_layer1)
    lstm_2 = LSTM(dim, activation='relu', kernel_initializer='random_uniform',bias_initializer='zeros',
                  return_sequences=False, name='lstm_layer2', recurrent_dropout=0.3)
    lstm_layer2 = lstm_2(bn_layer2)


    return Model(input, lstm_layer2)

file_name = './train_five_tuple_vector_data.json'
#file_name = './five_tuple_vector_data.json'
epochs = 50
batch_size = 512

df = pd.read_json(file_name)
df = df.drop(['Unnamed: 0'],axis=1)

data_pairs= create_pairs(df)
left,right,label = process_data(data_pairs)

num = left.shape[0]

left = left.reshape(num,70,5)
right = right.reshape(num,70,5)

base_network = create_base_network(HID_DIM)

input_left = Input(shape=(SQUENCE_NUM,VECTOR_UNIT))
input_right = Input(shape=(SQUENCE_NUM,VECTOR_UNIT))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_left = base_network(input_left)
processed_right = base_network(input_right)
#print(processed_left.shape)
#print(processed_left[0].shape)

distance = Lambda(Euclidean_distance,output_shape=eucl_dist_output_shape)([processed_left, processed_right])
#distance = Lambda(function=lambda x: Euclidean_distance(x[0], x[1]),output_shape=lambda x: (x[0], 1))([processed_left, processed_right])


model = Model([input_left, input_right], distance)
model.load_weights('./Euclidean_distance_epochs_100batch_size_512_lr0001+00001_correct.h5')
#model = multi_gpu_model(model,gpus =3)   #用gpu训练

model.summary()

model_path = "./Euclidean_distance_epochs_"+str(epochs)+"batch_size_"+str(batch_size)+"_lr0001+00001+000001_correct.h5"
#model_path = "./model/cosine_distance_epochs_"+str(epochs)+"batch_size_"+str(batch_size)+"_005.h5"
checkpoint = ModelCheckpoint(filepath=model_path,verbose=0,save_best_only=True)

# train
#rms = RMSprop()
#adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#adam = Adam(lr=0.001,clipvalue=0.1)
#adam = Adam(lr=0.00001,clipnorm=1.0)
adam = Adam(lr=0.00001)
model.compile(loss=contrastive_loss, optimizer=adam, metrics=[accuracy])
#model.compile(loss=contrastive_loss, optimizer=adam)
'''
history=model.fit([left, right], label,
          batch_size=128,
          epochs=epochs,verbose=2,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
'''
history=model.fit([left, right], label,
          batch_size=batch_size,
          #shuffle=True,
          shuffle=False,
          validation_split=0.2,
          epochs=epochs,verbose=1,callbacks=[checkpoint])


plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plot_train_history(history, 'loss', 'val_loss')
plt.subplot(1, 2, 2)
plot_train_history(history, 'accuracy', 'val_accuracy')
plt.show()