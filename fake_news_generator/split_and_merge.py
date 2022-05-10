#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import random
import math
import itertools


# In[15]:


SENTENCE_SEP = "।"
Inppath = "mergedclean.csv"
Outpath = "./fake_output.csv"
ROWS = 9999999 # put 999999 for all data

Inppath = input("Enter input path of csv: ")


# In[16]:


def pushnewlines(para,delim):
    if(para.count("\n")>5): return para
    plist = para.split(delim)
    paranew = ""
    for sent in plist:
        tp = 2-int(math.log2(random.randint(1,7)))
        paranew +=delim+"\n"*tp + sent
    return paranew


# In[17]:


dfjson = pd.read_csv(Inppath)
dfjson = dfjson.head(ROWS).copy()


# In[18]:


df = dfjson[["body","heading"]].copy()
df["label"] = 0
df.head(2)


# In[19]:


nrows = df.shape[0]
df.shape,nrows


# In[20]:


df["body"] = df["body"].apply(pushnewlines,delim = SENTENCE_SEP)


# In[ ]:





# In[21]:


def samplepara(para):
    plist = para.split("\n\n")
    return random.sample(plist,len(plist)//3)
    
    
def mixture(dfseries,index):
    n = dfseries.shape[0]
    ids = random.sample(range(n),3) + [index]
    id1,*idrem = ids
    mix = [samplepara(dfseries.iloc[i]) for i in idrem]
    mix = list(itertools.chain(*mix))
    random.shuffle(mix)
    
    # separately append in beginning
    mix = samplepara(dfseries.iloc[id1]) + mix
    return "".join(mix)
    


# In[ ]:





# In[22]:


# dffake = df.copy()
# dffake["label"] = 1
# for i in range(dffake.shape[0]):
#     dffake.iat[i,0] = mixture(df["body"],i)


# In[23]:


dffake = df.copy().to_dict()
for i in range(df.shape[0]):
    dffake["body"][i] = mixture(df["body"],i)
    dffake["label"][i] = 1


# In[24]:


dffake = pd.DataFrame(dffake)


# In[ ]:





# In[25]:


finalDf = pd.concat((df,dffake))

finalDf.to_csv(Outpath)


# In[26]:


finalDf


# In[27]:


# print(mixture(df["body"],index =0))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




