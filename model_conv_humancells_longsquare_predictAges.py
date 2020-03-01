#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Flatten,Dropout,MaxPooling1D, Input, Dense, LSTM, RepeatVector,TimeDistributed, Flatten,Bidirectional,Conv1D, GRU, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Concatenate, concatenate


# In[2]:


import pickle 

def read_pickleFile(filename):
    # returns dump pickle file object
    with open (filename,'rb') as pf:
        new_data=pickle.load(pf, encoding="bytes")
    return new_data


# In[3]:


import pickle
filename='human_cells_170_stimulus.pkl'
new_data=read_pickleFile(filename)

i=new_data[str.encode('i')]
t=new_data[str.encode('t')]
v=new_data[str.encode('v')]
ids=new_data[str.encode('id')]


# In[4]:


import pandas as pd
import numpy as np
# stimulus_df = pd.DataFrame(i[0],columns=["stimulus"])
response_df = pd.DataFrame(v[0],columns=["response"])
time_df = pd.DataFrame(t[0],columns=["time"])
for i_index in range(1,len(i)):
    response_df=response_df.append(pd.DataFrame(v[i_index],columns=["response"]),ignore_index=True)
#     stimulus_df=stimulus_df.append(pd.DataFrame(i[i_index],columns=["stimulus"]),ignore_index=True)
    time_df=time_df.append(pd.DataFrame(t[i_index],columns=["time"]),ignore_index=True)

df=time_df.join(response_df)
df


# In[5]:


df["response"]=df["response"].astype(np.int32)
df["response"].dtypes


# In[6]:



subsample=20
batch_size = 256
# sequence_length = 2525
sequence_length = 50500
x_data = df["response"].values[0::,]
# len(y_data)
y_data = df["response"].values[0::,]
y_data = y_data.reshape(-1, 1)
x_data = x_data.reshape(-1, 1)
print(type(y_data))
print("input Shape:", x_data.shape)
# y_data.reshape((400750,1))
# y_data
print("output Shape:", y_data.shape)

num_data = len(x_data)

num_train = 300*sequence_length

num_test = num_data - num_train
x_train = x_data[0:num_train]
x_test = x_data[num_train:]
# len(x_train) + len(x_test)
# x_train

y_train = y_data[0:num_train]
y_test = y_data[num_train:]
# len(y_train) + len(y_test)

num_x_signals = x_data.shape[1]
num_y_signals=y_data.shape[1]



# In[7]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler

import random
# from tf.keras.util import normalize
def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """
    idx=[]
    y_scaler = StandardScaler()
    for i in range(0,num_train,sequence_length):
        idx.append(i)
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.int32)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.int32)
        
        
        # Fill the batch with random sequences of data.
        idxx=0
        remove_list=[]
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            
            idxx = random.choice(idx)
            idx.remove(idxx)
            remove_list.append(idxx)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train[idxx:idxx+sequence_length]
            y_batch[i] = y_train[idxx:idxx+sequence_length]
            y_batch[i] =  y_scaler.fit_transform(y_batch[i])
            x_batch[i] =  y_scaler.fit_transform(x_batch[i])
#             y_batch[i]=y_batch[i][0]/100
#             x_batch[i]=x_batch[i][0]/100
        
        for i in remove_list:
            idx.append(i)
            
#         subsequences = 1
#         timesteps = x_batch.shape[1]//subsequences
#         x_batch= x_batch.reshape((x_batch.shape[0], timesteps, x_batch.shape[2]))
#         y_batch=y_batch.reshape((y_batch.shape[0],y_batch.shape[1],y_batch.shape[2]))
        subsequences = 5
        timesteps = x_batch.shape[1]//subsequences
        x_batch= x_batch.reshape((x_batch.shape[0],subsequences, timesteps, x_batch.shape[2]))
        y_batch=y_batch.reshape((y_batch.shape[0],subsequences, timesteps,y_batch.shape[2]))
        x_batch=x_batch.astype(np.int32)
        y_batch=y_batch.astype(np.int32)
        yield (x_batch, y_batch)


# In[8]:


def test_batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of testing-data.
    """
    test_idx=[]
    y_scaler = StandardScaler()
    for i in range(0,num_test,sequence_length):
        test_idx.append(i)
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_test_batch = np.zeros(shape=x_shape, dtype=np.int32)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_test_batch = np.zeros(shape=y_shape, dtype=np.int32)
        
        
        # Fill the batch with random sequences of data.
        for i in range(batch_size):
#         for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            
            idxx = random.choice(test_idx)
            test_idx.remove(idxx)
            
            # Copy the sequences of data starting at this index.
            x_test_batch[i] = x_test[idxx:idxx+sequence_length]
            y_test_batch[i] = y_test[idxx:idxx+sequence_length]
            y_test_batch[i] =  y_scaler.fit_transform(y_test_batch[i])
            x_test_batch[i] =  y_scaler.fit_transform(x_test_batch[i])
#             y_test_batch[i]=y_test_batch[i]/100
#             x_test_batch[i]=x_test_batch[i]/100
        
        for i in range(0,num_test,sequence_length):
            test_idx.append(i)
            
        subsequences = 5
        timesteps = x_test_batch.shape[1]//subsequences
        x_test_batch= x_test_batch.reshape((x_test_batch.shape[0],subsequences, timesteps, x_test_batch.shape[2]))
        y_test_batch=y_test_batch.reshape((y_test_batch.shape[0],subsequences, timesteps, y_test_batch.shape[2]))
        x_test_batch=x_test_batch.astype(np.int32)
        y_test_batch=y_test_batch.astype(np.int32)
        yield (x_test_batch, y_test_batch)


# In[16]:


test_generator = test_batch_generator(batch_size=4,
                            sequence_length=sequence_length)


# In[17]:


generator = batch_generator(batch_size=16,
                            sequence_length=sequence_length)


# In[18]:


x_batch, y_batch = next(generator)
print(x_batch.shape)


# In[19]:


x_test_batch,y_test_batch=next(test_generator)


# In[20]:


x_test_batch.shape


# In[21]:


validation_data = (x_test_batch,
                  y_test_batch)


# In[22]:



from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


# In[20]:


path_checkpoint = 'checkpoint_conv1d_int32.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)


# In[21]:


callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=1, verbose=1)


# In[22]:


callback_tensorboard = TensorBoard(log_dir='./logs/checkpoint_conv1d_int32/',
                                   histogram_freq=0,
                                   write_graph=False)


# In[23]:


callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=0.001,
                                       patience=0,
                                       verbose=1)


# In[24]:


callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]


# In[25]:


x_batch.shape


# In[26]:


subsequences=5
timesteps=10100


# In[24]:


from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector,TimeDistributed, Flatten,Bidirectional,Conv1D, GRU, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Concatenate, concatenate
from keras.initializers import RandomNormal, lecun_normal

def get_model_conv():
    num_units=200
#     model = Sequential()
    input_layer = Input(shape=(None,x_batch.shape[2],x_batch.shape[3]))
    # encoder
    conv=TimeDistributed(Conv1D(filters=1, kernel_size=128, padding="same",activation='relu'))(input_layer)
#     conv=TimeDistributed(Flatten())(conv)
    conv=TimeDistributed(Reshape((-1,1)))(conv)
    
#     conv=TimeDistributed(Dense(1))(conv)
    #     conv=Reshape((x_batch.shape[1]*x_batch.shape[2], 1))(conv)
    #     lstm1 = Bidirectional(LSTM(return_sequences=False, units=num_units))(conv)
    #     lstm2 = Bidirectional(LSTM(return_sequences=False, units=num_units))(conv)
    #     encoded = concatenate([lstm1, lstm2], axis=-1)

    #     rep_vec = RepeatVector(x_batch.shape[1]*x_batch.shape[2])(encoded)

    # #     # layer 2
    #     lstm3 = Bidirectional(LSTM(return_sequences=True, units=num_units))(rep_vec)
    #     lstm4 = Bidirectional(LSTM(return_sequences=True, units=num_units))(rep_vec)
    #     merge2 = concatenate([lstm3, lstm4], axis=-1)
    dense=TimeDistributed(Dense(500))(conv)
    dense=TimeDistributed(Dense(1))(dense)
#     output=Reshape((x_batch.shape[1],x_batch.shape[2],x_batch.shape[3]))(dense)
#     dense=Dense(1,activation="relu")(merge2)
    
    autoencoder = Model(input_layer, dense)

    print(autoencoder.summary())
    return autoencoder



# In[25]:


model = get_model_conv()

model.compile(loss='mean_squared_error', optimizer='adam')


# In[31]:


# # model.evaluate(x_test_neuron, y_test_neuron_scaled)
# # %%time
his=model.fit_generator(generator=generator,
                    epochs=100,
                    steps_per_epoch=100,
                       validation_data=validation_data,callbacks=callbacks)
his


# In[167]:


his


# In[136]:



from matplotlib import pyplot
ker=[]
for layer in model.layers:
    if len(layer.get_weights())>0:
        print(layer.get_weights()[0].shape)
        ker.append(layer.get_weights()[0])
    print(layer.get_config(), layer.get_weights())


# In[41]:


import matplotlib.pyplot as plt
kernal1=ker[0].reshape(128,)
plt.plot(kernal1, c='r')
plt.show()


# In[42]:


model.layers[3].output


# In[43]:


model1 = Model(inputs=model.inputs, outputs=model.layers[1].output)


# In[44]:


from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
scaler = StandardScaler()

i=0
y=x_test[sequence_length*i:(sequence_length*i)+sequence_length]
    # x=x_test[50500]
    # print("Min:", np.min(y))
    # print("Max:", np.max(y))
y=scaler.fit_transform(y)
# print("Min:", np.min(y))
# print("Max:", np.max(y))
len(y)
y=y.reshape(1,5,10100,1)
y_to=y.reshape(sequence_length,)
plt.plot(y_to, c='r')
plt.show()
predicted = model1.predict(y)
    # print("Min:", np.min(predicted))
    # print("Max:", np.max(predicted))
predicted=predicted.reshape(sequence_length,)
plt.plot(predicted,c='b',label='predicted')
plt.show()


# In[33]:


print(x_train.shape)
x_train_new=x_train.reshape(300,5,10100,1)


# In[34]:


print(x_test.shape)
x_test_new=x_test.reshape(45,5,10100,1)


# In[35]:


num_train_samples=300
num_test_samples=45


# In[36]:


sub_x_train=x_train_new[0:num_train_samples:,:]


# In[37]:


sub_x_test=x_test_new[0:num_test_samples:,:]


# In[38]:


scalar=StandardScaler()


# In[39]:



sub_x_train=sub_x_train.reshape(num_train_samples,5,10100,1)
# sub_x_train=scalar.fit_transform(sub_x_train)
sub_x_test=sub_x_test.reshape(num_test_samples,5,10100,1)
# sub_x_test=scalar.fit_transform(sub_x_test)


# In[40]:


test_idx=[]
map_idx_specimenId={}
count=0
for i in range(0,num_data,sequence_length):
    test_idx.append(i)
    
    map_idx_specimenId[ids[count]]=i
    count=count+1
len(test_idx)


# In[41]:


import pickle
filename='humancell_metadata_age_gender'
meta_data=read_pickleFile(filename)
ages=[]
for idx in ids:
    for k,v in meta_data.items():
        if idx == k:
            for k,v in v.items():
                if k==str.encode('age'):
                    str_age=v.decode('ascii')
                    if str_age=='unknown':
                        ages.append(42)
                    else:
                        ages.append(int(str_age.split(" ")[0]))
print(len(ages))
ages=np.asarray(ages)


# In[42]:


sub_y_train_ages=ages[0:num_train_samples]
sub_y_test_ages=ages[300:300+num_test_samples]
len(sub_y_test_ages)


# In[43]:


sub_y_train_ages=sub_y_train_ages.reshape(-1,1)
sub_y_test_ages=sub_y_test_ages.reshape(-1,1)


# In[44]:


y_scaler = MinMaxScaler()
sub_y_train_ages=y_scaler.fit_transform(sub_y_train_ages)
sub_y_test_ages=y_scaler.fit_transform(sub_y_test_ages)


# In[26]:


model.evaluate_generator(generator=test_generator,steps=4)


# In[27]:


path_checkpoint = 'checkpoint_conv1d_int32.keras'
model.load_weights(path_checkpoint)


# In[28]:


model.evaluate_generator(generator=test_generator,steps=4)


# In[30]:


model1 = Model(inputs=model.inputs, outputs=model.layers[3].output)


# In[46]:


feature_maps_train = model1.predict(sub_x_train)
feature_maps_test = model1.predict(sub_x_test)
print(feature_maps_train.shape)
print(feature_maps_test.shape)


# In[45]:


import gc
gc.collect()


# In[51]:


feature_maps_train=feature_maps_train.reshape(num_train_samples,subsequences*timesteps,500)
feature_maps_test=feature_maps_test.reshape(num_test_samples,subsequences*timesteps,500)


# In[52]:


feature_maps_train=feature_maps_train[:,:1].reshape(num_train_samples,500)
feature_maps_test=feature_maps_test[:,:1].reshape(num_test_samples,500)


# In[38]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(feature_maps)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


# In[53]:


for i in range(4):
    model.layers[i].trainable = False
#     print(i)


# In[71]:



input_layer = Input(shape=(500))
# input_layer= model.layers[3].output

ll=Dense(300,activation="relu",kernel_regularizer=l2(0.01))(input_layer)
# ll=Dropout(0.02)(ll)
ll=Dense(300,activation="relu",kernel_regularizer=l2(0.01))(ll)
ll=Dense(1)(ll)
new_model = Model(inputs=input_layer,outputs=ll)


# In[79]:


new_model.summary()


# In[73]:



new_model.compile(loss='mean_squared_error', optimizer='adam')


# In[74]:


validation_data = (feature_maps_test,
                  sub_y_test_ages)


# In[75]:


# model.evaluate(x_test_neuron, y_test_neuron_scaled)
# %%time
new_model.fit(feature_maps_train,sub_y_train_ages,
                    epochs=3000,validation_data=validation_data,callbacks=callbacks)


# In[76]:


predictions=new_model.predict(feature_maps_test)


# In[77]:


real_predicted_ages=y_scaler.inverse_transform(predictions)
real_predicted_ages


# In[78]:


true_ages=y_scaler.inverse_transform(sub_y_test_ages)
true_ages


# In[ ]:




