#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import io
from collections import OrderedDict
from pprint import pprint


# In[2]:


def gen_category_list(categories_to_use):
    categories = []
    
    for i, cat in enumerate(categories_to_use):
        category_dict = OrderedDict()
        category_dict['id'] = i + 1
        category_dict['name'] = cat
        categories.append(category_dict)
    
    return categories


# In[3]:


def gen_image_annotation_lists(data_dir, categories_to_use, max_occlusion_level=1):
    # Some constants
    csv_columns = ['bbox_left', 'bbox_top', 'bbox_width', 'bbox_height', 'score', 'object_category', 'truncation', 'occlusion']
    
    object_categories = [
        'ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'
    ]
    
    image_dir = os.path.join(data_dir, 'images')
    annotation_dir = os.path.join(data_dir, 'annotations')
    
    # lists to return
    images = []
    annotations = []
    
    annotation_id = 0
    
    for image_id, image_file in enumerate(tqdm(os.listdir(image_dir))):
        # Image
        file_basename, _ = os.path.splitext(image_file)
        image_path = os.path.join(image_dir, image_file)
        image = io.imread(image_path)
        h, w, _ = image.shape

        image_dict = OrderedDict()
        image_dict['file_name'] = image_file
        image_dict['height'] = h
        image_dict['width'] = w
        image_dict['id'] = image_id
        
        images.append(image_dict)

        # Annotation
        annotation_file = file_basename + '.txt'
        annotation_path = os.path.join(annotation_dir, annotation_file)
        objects = pd.read_csv(annotation_path, names=csv_columns)

        for i, obj in objects.iterrows():
            cat = object_categories[obj['object_category']]
            if not cat in categories_to_use:
                continue
                
            occ = obj['occlusion']
            if occ > max_occlusion_level:
                continue

            x, y = obj['bbox_left'], obj['bbox_top']
            bh, bw = obj['bbox_height'], obj['bbox_width']

            annotation_dict = OrderedDict()
            annotation_dict['area'] = 64 ** 2
            annotation_dict['iscrowd'] = 0
            annotation_dict['image_id'] = image_id
            annotation_dict['bbox'] = [float(x), float(y), float(bw), float(bh)]
            annotation_dict['category_id'] = categories_to_use.index(cat) + 1
            annotation_dict['id'] = annotation_id
            
            annotations.append(annotation_dict)
            annotation_id += 1
    
    return images, annotations


# In[4]:


def convert_label2json(data_dir, categories_to_use, out_path):
    print('Converting {} ...'.format(data_dir))
    category_list = gen_category_list(categories_to_use)
    image_list, annotation_list = gen_image_annotation_lists(data_dir, categories_to_use)
    
    label_dict = OrderedDict()
    label_dict['images'] = image_list
    label_dict['annotations'] = annotation_list
    label_dict['categories'] = category_list
    
    if out_path is not None:
        with open(out_path, 'w') as f:
            json.dump(label_dict, f)
    
    return label_dict


# In[5]:


#categories_to_use = [
#    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
#]

#convert_label2json('data/VisDrone2018-DET-val', categories_to_use, 'val.json')
#convert_label2json('data/VisDrone2018-DET-train', categories_to_use, 'train.json')


# In[6]:


categories_to_use = [
    'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

convert_label2json('data/VisDrone2018-DET-val', categories_to_use, 'val.json')
convert_label2json('data/VisDrone2018-DET-train', categories_to_use, 'train.json')
