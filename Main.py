#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn

import seaborn as sns

np.random.seed(1) # set a seed so that the results are consistent


# In[12]:


import tensorflow as tf


# In[13]:


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# In[15]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[17]:


from mlxtend.plotting import plot_decision_regions


# In[18]:


from sklearn.datasets import make_gaussian_quantiles
# Construct dataset
X1, y1 = make_gaussian_quantiles(cov=3.,
                                 n_samples=10000, n_features=2,
                                 n_classes=2, random_state=1)

y1 = y1.reshape(-1,1)


# In[19]:


total = np.concatenate( (X1,y1) , axis = 1)
df = pd.DataFrame( total , columns = ['f1' , 'f2' , 'Y']  )


# In[20]:


sns.set()


# In[21]:


sns.scatterplot( x = 'f1' , y = 'f2' , hue = 'Y' , style = 'Y' , data = df )


# In[22]:


linear_clf = sklearn.linear_model.LogisticRegressionCV()


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_tr , X_test , Y_tr , Y_test =     train_test_split( X1 , y1 , 
                     stratify = y1 , 
                     random_state = 3,  
                     test_size = 0.1)


# In[25]:


linear_clf.fit(X_tr,Y_tr)


# In[26]:


plot_decision_regions( X_test , Y_test.reshape(1000) , 
                       clf = linear_clf , 
                       legend = 2)

plt.show()


# In[27]:


from keras import models
from keras import layers


# In[28]:


def create_model(  in_size ,out_size , hidden_layers ):
    model = models.Sequential()
    model.add( layers.Dense(hidden_layers[0] , activation = 'relu' ,  input_shape = (in_size,)))
    
    for n_cells in hidden_layers[1:-1]:
        model.add( layers.Dense( n_cells , activation = 'relu' ) )
        
    model.add( layers.Dense( out_size , activation = 'sigmoid' ) )
    
    return model


# In[29]:


model = create_model( X_tr.shape[1] , hidden_layers = [3,3,3] , out_size = Y_tr.shape[1] )

model.compile(
                optimizer = 'adam',
                loss = 'mse',
                metrics = ['accuracy']
)


# In[34]:


model.fit(X_tr,Y_tr , epochs = 20)
plot_decision_regions( X_test , Y_test.squeeze() , clf = model , legend = 2 )


# In[ ]:




