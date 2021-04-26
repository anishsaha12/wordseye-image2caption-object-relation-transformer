#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import datetime
import json
import requests
import shutil
from nltk.tokenize import RegexpTokenizer


# In[2]:


DOWNLOAD_IMAGES = True


# In[3]:


def createSafeDir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp)
        
def preProcessCaption(caption):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(caption.lower())


# In[4]:


# code from https://stackoverflow.com/questions/18470627/how-do-i-remove-the-microseconds-from-a-timedelta-object
def cm(delta):
    return delta - datetime.timedelta(microseconds=delta.microseconds)


# In[5]:


createSafeDir("created_data")
if DOWNLOAD_IMAGES:
    createSafeDir("created_data/val")
    createSafeDir("created_data/train")


# In[6]:


data_root = "wordseye_data_post2013_okay"
data_splits = ["train", "test", "dev"]


# In[7]:


datapoint_dict_list = []
global_idx = 0

start_time = datetime.datetime.now()

for data_split in data_splits:
    
    print("***********************************************************************************************", flush = True)
    print("starting data_split: {}".format(data_split), flush = True)
    total_count = 0
    failed_count = 0
    
    file_name = data_split + ".txt"
    file_path = data_root + "/" + file_name
    f = open(file_path, 'r', encoding="utf8")
    Lines = f.readlines()
    count = 0
    for line in Lines:
        if(total_count%100 == 0):
            print("intermediate stats -- total_count: {}, failed_count: {}, success_count: {}, time_elapsed: {}".format(total_count, failed_count, total_count - failed_count, cm(datetime.datetime.now() - start_time)), flush = True)
            print("", flush = True)
        total_count = total_count + 1
        split_line = line.split("\t")
        if type(split_line) != list:
            print("type_error")
            
        if len(split_line) != 3:
            print("len error")
                    
        datapoint_id = split_line[0]
        
        if datapoint_id.isnumeric() == False:
            print("datapoint id is not numeric")
        datapoint_link = split_line[1]
        datapoint_caption = split_line[2]
        
        detected_index = None
        
        if "\n" in datapoint_caption:
            detected_index = datapoint_caption.rindex("\n")
            if detected_index + 1 != len(datapoint_caption):
                print("something's wrong")
            datapoint_caption = datapoint_caption[:detected_index - 1 + 1]
        
        datapoint_caption = datapoint_caption.strip()

        # Code from https://stackoverflow.com/questions/2486145/python-check-if-url-to-jpg-exists
        r = requests.head(datapoint_link)
        if r.status_code != requests.codes.ok:
            failed_count = failed_count + 1
        else:

            url_parts = datapoint_link.split("/")
            my_filename = "_".join(datapoint_link.split("/")[-3:])
            
            my_filepath = None
            if data_split == "test" or data_split == "dev":
                my_filepath = "val"
            else:
                my_filepath = "train"
                
                
                
            my_split = None
            if data_split == "train" or data_split == "test":
                my_split = data_split
            elif data_split == "dev" and global_idx%2==0:
                my_split = "val"
            elif data_split == "dev" and global_idx%2==1:
                my_split = "restval"
                
            
            datapoint_dict = {
                                    "filepath": my_filepath,
                                    "sentids": [global_idx],
                                    "filename": my_filename,
                                    "imgid": global_idx,
                                    "split": my_split,
                                    "sentences": [{
                                        "tokens": preProcessCaption(datapoint_caption),
                                        "raw": datapoint_caption,
                                        "imgid": global_idx,
                                        "sentid": global_idx
                                    }],
                                    'wordseyeid': global_idx
                                    
            }
            datapoint_dict_list.append(datapoint_dict)
            
            
            if DOWNLOAD_IMAGES:
                r = requests.get(datapoint_link, stream=True)
                with open("created_data/" + my_filepath + "/" + my_filename, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)  

            
            
            global_idx = global_idx + 1
            
    f.close()
    
    print("ending " + data_split + " - stats -- total_count: {}, failed_count: {}, success_count: {}, time_elapsed: {}".format(total_count, failed_count, total_count - failed_count, cm(datetime.datetime.now() - start_time)), flush = True)
    print("***********************************************************************************************", flush = True)
    print("", flush = True)
    print("", flush = True)


# In[8]:


print("len(datapoint_dict_list): {}".format(len(datapoint_dict_list)))
content = {
    "dataset": "wordseye",
    "images": datapoint_dict_list
}

f2 = open("created_data/dataset_wordseye.json", "w")
json.dump(content, f2, indent = 4)
f2.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




