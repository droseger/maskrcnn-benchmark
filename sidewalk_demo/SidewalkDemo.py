#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

import random
from pathlib import Path


# In[2]:


from maskrcnn_benchmark.config import cfg
from predictor import SidewalkDemo


# In[3]:


pylab.rcParams['figure.figsize'] = 20, 12


# In[4]:


config_file = "/home/daniel.rose/maskrcnn-benchmark/configs/sidewalk.fpn.bs1-final.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])


# In[5]:


def load():
    path = Path('/home/daniel.rose/maskrcnn-benchmark/datasets/sidewalk/test/JPEGImages/')
    file = random.choice(list(path.glob('*.jpg')))
    pil_image = Image.open(file).convert("RGB")
    pil_image = pil_image.resize((800,800))
    pil_image = pil_image.convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")


# In[6]:


default = SidewalkDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)


# In[7]:


jitter = SidewalkDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
    brightness = (5,5),
    # hue = 0.3
)


# In[13]:


image = load()
imshow(image)


# In[14]:


predictions = default.run_on_opencv_image(image)
imshow(predictions)


# In[15]:


predictions = jitter.run_on_opencv_image(image)
imshow(predictions)


# In[ ]:





