import tensorflow as tf
import numpy as np
from Layer_functions import *
from utils import *
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import uproot
import pandas as pd
from tqdm import tqdm

tf.config.threading.set_inter_op_parallelism_threads(20)
tf.config.threading.set_intra_op_parallelism_threads(20)

#file = uproot.open("/u/jscharf/Documents/Daten/simple model/Speicher/gps3_E20_spread0.3GeV_halfBox8mmAir_t25.1_nch50_d5_pixelThr82_noise20_stepLength1.root:CaloOutputWriter/Frames")
# print('Diese Branches sind in dem Baum:', file.typenames())
#Einladen der Branches in Pandas Series
#rows =file["row"].array(library='np')
#columns= file["column"].array(library='np')
#nHits=file["nHits"].array(library='np')
#lanes= file["lane"].array(library='np')


def train_gen():
    file = uproot.open("/u/jscharf/Documents/Daten/simple model/Speicher/gps3_E20_spread0.3GeV_halfBox8mmAir_t25.1_nch50_d5_pixelThr82_noise20_stepLength1.root:CaloOutputWriter/Frames")
    rows = file["row"].array(library='np')
    columns = file["column"].array(library='np')
    nHits = file["nHits"].array(library='np')
    lanes = file["lane"].array(library='np')
    file_2 = uproot.open("/u/jscharf/Documents/Daten/simple model/Speicher/gps6_E20_spread0.3GeV_halfBox8mmAir_t25.1_nch50_d5_pixelThr82_noise20_stepLength1.root:CaloOutputWriter/Frames")
    rows_2 = file_2["row"].array(library='np')[:20000]
    columns_2 = file_2["column"].array(library='np')[:20000]
    nHits_2 = file_2["nHits"].array(library='np')[:20000]
    lanes_2 = file_2["lane"].array(library='np')[:20000]

    rows = rows[:14000]
    columns = columns[:14000]
    nHits = nHits[:14000]
    lanes = lanes[:14000]
    rows_2 = rows_2[:14000]
    columns_2 = columns_2[:14000]
    nHits_2 = nHits_2[:14000]
    lanes_2 = lanes_2[:14000]

    assert nHits.shape == nHits_2.shape
    for i in range(nHits.shape[0]+nHits_2.shape[0]):
        if i % 2 == 0:
            b=i//2
            lane = lanes[b]
            row = rows[b]
            col = columns[b]
            a = 1
        else:
            b=i//2
            lane = lanes_2[b]
            row = rows_2[b]
            col = columns_2[b]
            a = 0
        lane, row, col = change_to_real_coordinates_for_one_event(lane, row, col)
        lane = tf.expand_dims(tf.convert_to_tensor(lane, dtype=tf.int32), axis=-1)
        row = tf.expand_dims(tf.convert_to_tensor(row, dtype=tf.int32), axis=-1)
        col = tf.expand_dims(tf.convert_to_tensor(col, dtype=tf.int32), axis=-1)
        data = tf.concat([lane, row, col], axis=-1)
        yield data, a

def test_gen():
    file = uproot.open("/u/jscharf/Documents/Daten/simple model/Speicher/gps3_E20_spread0.3GeV_halfBox8mmAir_t25.1_nch50_d5_pixelThr82_noise20_stepLength1.root:CaloOutputWriter/Frames")
    rows = file["row"].array(library='np')
    columns = file["column"].array(library='np')
    nHits = file["nHits"].array(library='np')
    lanes = file["lane"].array(library='np')
    file_2 = uproot.open("/u/jscharf/Documents/Daten/simple model/Speicher/gps6_E20_spread0.3GeV_halfBox8mmAir_t25.1_nch50_d5_pixelThr82_noise20_stepLength1.root:CaloOutputWriter/Frames")
    rows_2 = file_2["row"].array(library='np')[:20000]
    columns_2 = file_2["column"].array(library='np')[:20000]
    nHits_2 = file_2["nHits"].array(library='np')[:20000]
    lanes_2 = file_2["lane"].array(library='np')[:20000]

    rows = rows[18000:]
    columns = columns[18000:]
    nHits = nHits[18000:]
    lanes = lanes[18000:]
    rows_2 = rows_2[18000:]
    columns_2 = columns_2[18000:]
    nHits_2 = nHits_2[18000:]
    lanes_2 = lanes_2[18000:]

    assert nHits.shape == nHits_2.shape
    for i in range(nHits.shape[0]+nHits_2.shape[0]):
        if i % 2 == 0:
            b=i//2
            lane = lanes[b]
            row = rows[b]
            col = columns[b]
            a = 1
        else:
            b=i//2
            lane = lanes_2[b]
            row = rows_2[b]
            col = columns_2[b]
            a = 0
        lane, row, col = change_to_real_coordinates_for_one_event(lane, row, col)
        lane = tf.expand_dims(tf.convert_to_tensor(lane, dtype=tf.int32), axis=-1)
        row = tf.expand_dims(tf.convert_to_tensor(row, dtype=tf.int32), axis=-1)
        col = tf.expand_dims(tf.convert_to_tensor(col, dtype=tf.int32), axis=-1)
        data = tf.concat([lane, row, col], axis=-1)
        yield data, a


def valid_gen():
    file = uproot.open("/u/jscharf/Documents/Daten/simple model/Speicher/gps3_E20_spread0.3GeV_halfBox8mmAir_t25.1_nch50_d5_pixelThr82_noise20_stepLength1.root:CaloOutputWriter/Frames")
    rows = file["row"].array(library='np')
    columns = file["column"].array(library='np')
    nHits = file["nHits"].array(library='np')
    lanes = file["lane"].array(library='np')
    file_2 = uproot.open("/u/jscharf/Documents/Daten/simple model/Speicher/gps6_E20_spread0.3GeV_halfBox8mmAir_t25.1_nch50_d5_pixelThr82_noise20_stepLength1.root:CaloOutputWriter/Frames")
    rows_2 = file_2["row"].array(library='np')[:20000]
    columns_2 = file_2["column"].array(library='np')[:20000]
    nHits_2 = file_2["nHits"].array(library='np')[:20000]
    lanes_2 = file_2["lane"].array(library='np')[:20000]

    rows = rows[14000:18000]
    columns = columns[14000:18000]
    nHits = nHits[14000:18000]
    lanes = lanes[14000:18000]
    rows_2 = rows_2[14000:18000]
    columns_2 = columns_2[14000:18000]
    nHits_2 = nHits_2[14000:18000]
    lanes_2 = lanes_2[14000:18000]

    assert nHits.shape == nHits_2.shape
    for i in range(nHits.shape[0]+nHits_2.shape[0]):
        if i % 2 == 0:
            b=i//2
            lane = lanes[b]
            row = rows[b]
            col = columns[b]
            a = 1
        else:
            b=i//2
            lane = lanes_2[b]
            row = rows_2[b]
            col = columns_2[b]
            a = 0
        lane, row, col = change_to_real_coordinates_for_one_event(lane, row, col)
        lane = tf.expand_dims(tf.convert_to_tensor(lane, dtype=tf.int32), axis=-1)
        row = tf.expand_dims(tf.convert_to_tensor(row, dtype=tf.int32), axis=-1)
        col = tf.expand_dims(tf.convert_to_tensor(col, dtype=tf.int32), axis=-1)
        data = tf.concat([lane, row, col], axis=-1)
        yield data, a



ds_train = tf.data.Dataset.from_generator(train_gen, output_types=(tf.int32, tf.int32), output_shapes=((None, 3), ()))
ds_train_batch= ds_train.shuffle(40).padded_batch(20)
ds_valid = tf.data.Dataset.from_generator(valid_gen, output_types=(tf.int32, tf.int32), output_shapes=((None, 3), ()))
ds_valid_batch= ds_valid.shuffle(40).padded_batch(20)
ds_test = tf.data.Dataset.from_generator(test_gen, output_types=(tf.int32, tf.int32), output_shapes=((None, 3), ()))
ds_test_batch= ds_test.shuffle(40).padded_batch(20)

input=keras.Input(shape=(None, 3))
x=layer_GarNet(input,20,10,6)
x=layer_GarNet(x,20,10,6)
x=layer_GarNet(x,20,10,6)
x=layer_GarNet(x,20,10,6)
x=layers.GlobalMaxPooling1D()(x)
x=layers.Dense(30, activation='gelu')(x)
x=layers.Dense(30, activation='gelu')(x)
x=layers.Dense(30, activation='gelu')(x)
x=layers.Dense(15, activation='gelu')(x)
out=layers.Dense(2, activation='softmax')(x)
model= keras.Model(inputs=input, outputs=out)
# model.summary()


model.compile(optimizer='nadam', loss='sparse_categorical_crossentropy', metrics='accuracy')
# early_stopping_cb=keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
# checkpoint_cb=keras.callbacks.ModelCheckpoint('/u/jscharf/Documents/Daten/Modelle/GarNet2', save_best_only=True)
model.fit(ds_train_batch, validation_data=ds_valid_batch,  epochs=100)# callbacks=[early_stopping_cb, checkpoint_cb])

model.evaluate(ds_test_batch)
