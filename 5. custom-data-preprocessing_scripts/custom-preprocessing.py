#!/usr/bin/env python
# coding: utf-8

# In[9]:


import sys
import os
import datetime
import json
import requests
import shutil
from nltk.tokenize import RegexpTokenizer
import re


# In[10]:


def createSafeDir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp)
        
def preProcessCaption(caption):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(caption.lower())


# code from https://stackoverflow.com/questions/18470627/how-do-i-remove-the-microseconds-from-a-timedelta-object
def cm(delta):
    return delta - datetime.timedelta(microseconds=delta.microseconds)


# In[11]:


custom_dataset_dir = "custom_dataset"


# In[12]:


wordseye_number = 120000 + 1


# In[13]:


createSafeDir("created_data")
createSafeDir("created_data/train")

datapoint_dict_list = []
global_idx = 0

start_time = datetime.datetime.now()

print("***********************************************************************************************", flush = True)
print("starting", flush = True)
total_count = 0
failed_count = 0

count = 0
for datapoint in os.listdir(custom_dataset_dir):
    total_count = total_count + 1
    datapoint_id = wordseye_number
    datapoint_caption = "".join(re.split("[^a-zA-Z]*", datapoint.split('.')[0]))
    
    wordseye_filename = "ws-image-db_2021-4-15_" + str(wordseye_number) + ".jpg"
    datapoint_dict = {
                            "filepath": "train",
                            "sentids": [global_idx],
                            "filename": wordseye_filename,
                            "imgid": global_idx,
                            "split": "train",
                            "sentences": [{
                                "tokens": preProcessCaption(datapoint_caption),
                                "raw": datapoint_caption,
                                "imgid": global_idx,
                                "sentid": global_idx
                            }],
                            'wordseyeid': global_idx

    }
    
    shutil.copy(custom_dataset_dir + "/" + datapoint, "created_data/train/" + wordseye_filename)
    datapoint_dict_list.append(datapoint_dict)
    global_idx = global_idx + 1
    wordseye_number = wordseye_number + 1


print("ending - stats -- total_count: {}, failed_count: {}, success_count: {}, time_elapsed: {}".format(total_count, failed_count, total_count - failed_count, cm(datetime.datetime.now() - start_time)), flush = True)
print("***********************************************************************************************", flush = True)
print("", flush = True)
print("", flush = True)


print("len(datapoint_dict_list): {}".format(len(datapoint_dict_list)))
content = {
    "dataset": "wordseye",
    "images": datapoint_dict_list
}

f2 = open("created_data/dataset_custom_wordseye.json", "w")
json.dump(content, f2, indent = 4)
f2.close()


# In[14]:


print(wordseye_number)


# In[ ]:




