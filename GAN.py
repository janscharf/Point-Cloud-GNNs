import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from utils import *
from Layer_functions import *
import uproot
import os
import time

tf.config.threading.set_inter_op_parallelism_threads(50)
tf.config.threading.set_intra_op_parallelism_threads(50)
tf.random.set_seed(42)
BATCH_SIZE=10
SHUFFLE_SIZE=3*BATCH_SIZE
TOTAL_SIZE= 20000
EPOCHS = 2
STEP_SIZE = 100
MIN_VALXY = 300
MAX_VALXY = 700
MIN_VALZ = 5
MAX_VALZ = 20
assert STEP_SIZE % BATCH_SIZE == 0
assert TOTAL_SIZE % STEP_SIZE == 0
#Models

in1= keras.Input(shape=(None, 3))
x=layer_GarNet(in1,20,10,6)
x=layer_GarNet(x,20,10,6)
x=layer_GarNet(x,20,10,6)
out1=layer_GarNet(x,20,3,6)
generator = keras.Model(inputs=in1, outputs=out1, name='generator')
generator.summary()

in2= keras.Input(shape=(None, 3))
y=layer_GarNet(in2,20,10,6)
y=layer_GarNet(y,20,10,6)
y=layer_GarNet(y,20,10,6)
y=layer_GarNet(y,20,10,6)
y=layers.GlobalMaxPooling1D()(y)
y=layers.Dense(30, activation='gelu')(y)
y=layers.Dense(30, activation='gelu')(y)
y=layers.Dense(30, activation='gelu')(y)
y=layers.Dense(15, activation='gelu')(y)
out2=layers.Dense(1, activation='linear')(y)
discriminator = keras.Model(inputs=in2, outputs=out2, name='discriminator')
discriminator.summary()


discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics="binary_accuracy")
discriminator.trainable= False
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(loss="binary_crossentropy", optimizer="adam", metrics="binary_accuracy")

file = uproot.open("/u/jscharf/Documents/Daten/simple model/Speicher/gps3_E20_spread0.3GeV_halfBox8mmAir_t25.1_nch50_d5_pixelThr82_noise20_stepLength1.root:CaloOutputWriter/Frames") #load in root file
rows = file["row"].array(library="np") #acces Hits, rows, lanes/layers and columns
columns = file["column"].array(library="np")
lanes = file["lane"].array(library="np")
nHits = file["nHits"].array(library="np")

def get_distribution():
    N_pre, counts_pre = np.unique(nHits, return_counts=True) #Get the unique amount of Hits and how often they occur in the dataset
    max_val = np.amax(N_pre) #get the max amount of Hits
    total = nHits.shape[0] #total amount of Events to normalize the distribution
    ranged_arr = np.arange(max_val + 1) #create range from 0 to max in steps of 1
    mask = np.in1d(ranged_arr, N_pre) #create a Mask True/False if the  value of ranged is in a unique value of the amount of Hits
    counter = np.zeros(shape=ranged_arr.shape) #create final probability counter filled with zeros
    j=0 #indices for counts_pre, as it has a different shape to mask
    for i in range(mask.shape[0]): #iterate of length of mask
        if mask[i]: #see if the amount of Hits happend at least once, if not 0 stays
            counter[i]=counts_pre[j] #fill how often it occured
            j+=1 #to the next step of counts_pre
    return ranged_arr, counter, total #returns the range, probability and the total for normalization, if we do normalization now sum(distribution)!=1 since we have a lot of very small numbers we get a rounding issue

distribution_ranged, distribution_probability, total= get_distribution() #Establish global range, probability and total of nHit distribution in the real dataset for the nosie generator to use



print("We are working with " + str(EPOCHS) + " Epochs, a Dataset containgin " + str(TOTAL_SIZE) + " datapoints, a Step size of " + str(STEP_SIZE) +" so a Step includes " + str(int(TOTAL_SIZE/STEP_SIZE)) + " datapoints and a Batch size of " + str(BATCH_SIZE) + " so there are " + str(int(STEP_SIZE/BATCH_SIZE)) + " Batches per Step.")
loss_d_noise = []
loss_d_real = []
loss_g_noise = []
for epoch in range(EPOCHS):
    BEGIN_STEP = 0
    END_STEP = STEP_SIZE
    print("This is Epoch " + str(epoch + 1) + ". We are working with a Step size of " + str(STEP_SIZE) +" and a Batch size of " + str(BATCH_SIZE))
    for step in range(int(TOTAL_SIZE/STEP_SIZE)):
        #Data Input
        historyd_real = 0
        historyd_noise = 0
        historyg_noise = 0
        print("This is Step " + str(step + 1) + " of " + str(int(TOTAL_SIZE/STEP_SIZE)) + " steps.")
        def real_data_generator():
            """
            Generates the real data from loading in the root file, converting the coordinates and putting them into the graph shape.
            """
            step_rows = rows[BEGIN_STEP:END_STEP]
            step_columns = columns[BEGIN_STEP:END_STEP]
            step_lanes = lanes[BEGIN_STEP:END_STEP]
            step_nHits = nHits[BEGIN_STEP:END_STEP]

            for i in range(step_nHits.shape[0]): #iterate over all Events
                row = step_rows[i] #get row, col, lane/layer of the specifc Event
                col = step_columns[i]
                lane = step_lanes[i]
                lane, row, col = change_to_real_coordinates_for_one_event(lane, row, col) #convert to actual coordinates
                lane = tf.expand_dims(tf.convert_to_tensor(lane, dtype=tf.int32), axis=-1) #add dimension for concat
                row = tf.expand_dims(tf.convert_to_tensor(row, dtype=tf.int32), axis=-1)
                col = tf.expand_dims(tf.convert_to_tensor(col, dtype=tf.int32), axis=-1)
                real_data = tf.concat([lane, row, col], axis=-1) #concat z + y + x = (z,y,x)
                yield real_data, 1 #return the real data to a dataset
        #Create a Dataset for the real data from real_data_generator and batch it
        ds_real = tf.data.Dataset.from_generator(real_data_generator, output_types=(tf.int32, tf.int32), output_shapes=((None, 3), ()))
        ds_real_batch = ds_real.shuffle(SHUFFLE_SIZE).padded_batch(BATCH_SIZE)









        #Generate Fake Data


        def noise_generator_real():
            """
            Creates an infinit amount of graph noise data with the create_graph_noise function. This will be put into a Dataset and batched. Adds the real label 1 to every data point.
            """
            p=0
            while p<4*STEP_SIZE:
                num_points = np.random.choice(distribution_ranged, p=distribution_probability/total) #Get the num_points from the distribution
                graph_noise_yx = tf.random.uniform(shape=[num_points, 2], minval=MIN_VALXY, maxval=MAX_VALXY, dtype=tf.dtypes.int32, name='Generate x, y Noise') #get y,x coordinates for the noise data from a uniform random distribution in [0, 1024)
                graph_noise_z = tf.random.uniform(shape=[num_points, 1], minval=MIN_VALZ, maxval=MAX_VALZ, dtype=tf.dtypes.int32, name='Generate z Noise') #get z coordinates for the noise data from a uniform random distribution in [0, 24), since it has not the same size as x,y coordinates
                graph_noise = tf.concat([graph_noise_z, graph_noise_yx], axis=1) #combine z + y,x -> z,y,x
                unique, _ = tf.raw_ops.UniqueV2(x=graph_noise, axis=[0]) #filter all unique points, to check if we have the same points twice or more
                if unique.shape[0] == num_points:
                    p+=1
                    yield graph_noise, 1

        def noise_generator_fake():
            """
            Creates an infinit amount of graph noise data with the create_graph_noise function. This will be put into a Dataset and batched. Adds the fake label 0 to every data point.
            """
            p=0
            while p<STEP_SIZE:
                num_points = np.random.choice(distribution_ranged, p=distribution_probability/total) #Get the num_points from the distribution
                graph_noise_yx = tf.random.uniform(shape=[num_points, 2], minval=MIN_VALXY, maxval=MAX_VALXY, dtype=tf.dtypes.int32, name='Generate x, y Noise') #get y,x coordinates for the noise data from a uniform random distribution in [0, 1024)
                graph_noise_z = tf.random.uniform(shape=[num_points, 1], minval=MIN_VALZ, maxval=MAX_VALZ, dtype=tf.dtypes.int32, name='Generate z Noise') #get z coordinates for the noise data from a uniform random distribution in [0, 24), since it has not the same size as x,y coordinates
                graph_noise = tf.concat([graph_noise_z, graph_noise_yx], axis=1) #combine z + y,x -> z,y,x
                unique, _ = tf.raw_ops.UniqueV2(x=graph_noise, axis=[0]) #filter all unique points, to check if we have the same points twice or more
                if unique.shape[0] == num_points:
                    graph_noise = tf.expand_dims(graph_noise, axis=0)
                    generated_noise = generator(graph_noise, training=False)
                    generated_noise = tf.squeeze(tf.convert_to_tensor(generated_noise), axis=0)
                    p+=1
                    yield generated_noise, 0


        #Put noise generator into a dataset and batch it:
        ds_noise_real = tf.data.Dataset.from_generator(noise_generator_real, output_types=(tf.int32, tf.int32), output_shapes=((None, 3), ()))
        ds_noise_real_batch = ds_noise_real.shuffle(SHUFFLE_SIZE).padded_batch(BATCH_SIZE)
        ds_noise_fake = tf.data.Dataset.from_generator(noise_generator_fake, output_types=(tf.int32, tf.int32), output_shapes=((None, 3), ()))
        ds_noise_fake_batch = ds_noise_fake.shuffle(SHUFFLE_SIZE).padded_batch(BATCH_SIZE)


















        #Training Loops
        def train(g_model, d_model, gan_model, ds_real_batch, ds_noise_real_batch, ds_noise_fake_batch):
            historyd_real = d_model.fit(ds_real_batch, epochs=1)
            loss_d_real.append(historyd_real.history['loss'])
            historyd_real = 0
            historyd_noise = d_model.fit(ds_noise_fake_batch, epochs=1)
            loss_d_noise.append(historyd_noise.history['loss'])
            historyd_noise = 0
            historyg_noise = gan_model.fit(ds_noise_real_batch, epochs=1)
            loss_g_noise.append(historyg_noise.history['loss'])
            historyg_noise = 0
        train(generator, discriminator, gan, ds_real_batch, ds_noise_real_batch, ds_noise_fake_batch)

        plot_loss(loss_d_real, loss_d_noise, loss_g_noise)
        BEGIN_STEP += STEP_SIZE
        END_STEP += STEP_SIZE


        #loop geht so for (x_noise, y_noise), (x_real, y_real) in zip(ds_noise_real_batch, ds_real_batch):
