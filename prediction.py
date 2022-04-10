#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  
#py.init_notebook_mode(connected=True)

## For scaling data 
from mlxtend.preprocessing import minmax_scaling 

# Tensorflow 
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from tensorflow.keras import layers
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import  plot_model
from keras import metrics


tf.get_logger().setLevel('ERROR')  

# In[2]:


md = keras.models.load_model('5.2_april.h5')


# In[3]:


optimizer1 = tf.keras.optimizers.Adam(0.007)


# In[4]:


md.compile(loss='binary_crossentropy', optimizer=optimizer1, metrics=['accuracy'])


# In[ ]:





# In[6]:


data = np.array([[4,250.0,72.0,29.0,126.0,30,1.5,60]])


# In[11]:


output =   md.predict(data)

out= np.multiply(output[0,0],100)
# In[10]:


print("you have a {}% chance of getting Diabetes".format(out))


# In[ ]:




