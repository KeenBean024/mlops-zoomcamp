#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pickle
import pandas as pd
import sys


# In[4]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[20]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[6]:


month = str(sys.argv[0])
year = str(sys.argv[1])

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet')


# In[7]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[9]:


print(y_pred.mean())


# In[12]:


# year = 2023
# month = 3
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[17]:


df_result = df[["ride_id"]]
df_result['predictions'] = y_pred


# In[19]:


df_result.to_parquet(
    'prediction_result.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)


# In[ ]:




