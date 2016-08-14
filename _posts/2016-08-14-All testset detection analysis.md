---
layout: post
title: The way how ConvNet see the images. CAM algorithm implementation
subtitle: by Veronika Yurchuk
bigimg: /img/path.jpg
show-avatar: false
---

### In this small post I decided to play more with SSD. My goal was to count all objects per all classes that Network was possible to detect. Also, it was interesting how class distribution change per different Neural Network's certainty.


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
%matplotlib inline
plt.rcParams['figure.figsize'] = (20, 20)
plt.rcParams.update({'font.size': 22})

import os
import operator
from collections import Counter

caffe_root = '/home/veronika/materials/cv/detection/caffe/examples/'  
# this file is expected to be in {caffe_root}/examples
import sys
#sys.path.append("/home/veronika/materials/cv/detection/caffe/build/tools/caffe")
import sys
sys.path.append('/home/veronika/materials/cv/detection/caffe/python')
sys.path.remove('/home/veronika/caffe/python')

import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = '/home/veronika/materials/cv/detection/git/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)
```

    /usr/local/lib/python2.7/dist-packages/matplotlib/font_manager.py:273: UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment.
      warnings.warn('Matplotlib is building the font cache using fc-list. This may take a moment.')





    <caffe.proto.caffe_pb2.LabelMap at 0x7f1ecb828938>




```python
def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames
```


```python
caffe.set_device(0)
caffe.set_mode_gpu()
model_def = '/home/veronika/materials/cv/detection/models_trained/VGGNet/deploy.prototxt'
model_weights = '/home/veronika/materials/cv/detection/models_trained/VGGNet/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
```


```python
labels = pd.read_csv("/home/veronika/materials/cv/cv_organizer/mydata/test_labels.csv",
                    sep = " ", header = None)
```


```python
def get_object_descr(path_to_test, conf_level):
    descr_total = {}
    for i in range(labels[0].shape[0]):
        #path_to_test = path_to_img
        image = caffe.io.load_image(os.path.join(path_to_test, labels[0][i]))
        transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = net.forward()['detection_out']
    
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_level]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(labelmap, top_label_indices)
        #descr = Counter(dict((i,top_labels.count(i)) for i in set(top_labels)))
        descr = Counter(top_labels)
        descr_total = descr + Counter(descr_total)
        print("Done!")
    return(descr_total)
```


```python
path_to_test = "/home/veronika/materials/cv/cv_organizer/mydata/test/"
object_desc_02 = get_object_descr(path_to_test, 0.2)
object_desc_04 = get_object_descr(path_to_test, 0.4)
object_desc_06 = get_object_descr(path_to_test, 0.6)
object_desc_08 = get_object_descr(path_to_test, 0.8)
```

This is a distribution per classes with 80% of model's certainty.


```python
object_desc_08
```




    Counter({u'aeroplane': 8,
             u'bicycle': 54,
             u'bird': 6,
             u'boat': 31,
             u'bottle': 3,
             u'bus': 5,
             u'car': 62,
             u'cat': 1,
             u'chair': 71,
             u'diningtable': 19,
             u'dog': 16,
             u'horse': 49,
             u'motorbike': 9,
             u'person': 2158,
             u'pottedplant': 30,
             u'sheep': 1,
             u'sofa': 14,
             u'train': 6,
             u'tvmonitor': 18})




```python
object_desc_02_df = pd.DataFrame(object_desc_02.items(), columns=['ClassType', 'Prob02'])
object_desc_04_df = pd.DataFrame(object_desc_04.items(), columns=['ClassType', 'Prob04'])
object_desc_06_df = pd.DataFrame(object_desc_06.items(), columns=['ClassType', 'Prob06'])
#object_desc_08_df = pd.DataFrame(object_desc_08.items(), columns=['ClassType', 'Prob08'])
```

The following table was actually my goal. I can see how many objects CNN found per class with different probability levels.


```python
totaldata = pd.merge(object_desc_02_df, object_desc_04_df, how='outer')
totaldata = pd.merge(totaldata, object_desc_06_df, how = "outer")
#totaldata = pd.merge(totaldata, object_desc_08_df, how = "outer")
totaldata
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClassType</th>
      <th>Prob02</th>
      <th>Prob04</th>
      <th>Prob06</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sheep</td>
      <td>17</td>
      <td>7</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bottle</td>
      <td>42</td>
      <td>9</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>horse</td>
      <td>101</td>
      <td>76</td>
      <td>64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bicycle</td>
      <td>168</td>
      <td>92</td>
      <td>61</td>
    </tr>
    <tr>
      <th>4</th>
      <td>motorbike</td>
      <td>50</td>
      <td>25</td>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cow</td>
      <td>17</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>bus</td>
      <td>21</td>
      <td>13</td>
      <td>11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>dog</td>
      <td>74</td>
      <td>46</td>
      <td>29</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cat</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>person</td>
      <td>4688</td>
      <td>3367</td>
      <td>2699</td>
    </tr>
    <tr>
      <th>10</th>
      <td>train</td>
      <td>41</td>
      <td>21</td>
      <td>11</td>
    </tr>
    <tr>
      <th>11</th>
      <td>diningtable</td>
      <td>70</td>
      <td>41</td>
      <td>26</td>
    </tr>
    <tr>
      <th>12</th>
      <td>aeroplane</td>
      <td>32</td>
      <td>18</td>
      <td>13</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sofa</td>
      <td>54</td>
      <td>39</td>
      <td>24</td>
    </tr>
    <tr>
      <th>14</th>
      <td>pottedplant</td>
      <td>146</td>
      <td>73</td>
      <td>45</td>
    </tr>
    <tr>
      <th>15</th>
      <td>tvmonitor</td>
      <td>85</td>
      <td>48</td>
      <td>36</td>
    </tr>
    <tr>
      <th>16</th>
      <td>chair</td>
      <td>357</td>
      <td>186</td>
      <td>112</td>
    </tr>
    <tr>
      <th>17</th>
      <td>bird</td>
      <td>42</td>
      <td>22</td>
      <td>14</td>
    </tr>
    <tr>
      <th>18</th>
      <td>boat</td>
      <td>102</td>
      <td>65</td>
      <td>45</td>
    </tr>
    <tr>
      <th>19</th>
      <td>car</td>
      <td>192</td>
      <td>119</td>
      <td>89</td>
    </tr>
  </tbody>
</table>
</div>



The next step was dataframe transformation, that is needed for plotting barplots using ggplot library.
More details are in the next post.


```python
totaldata_melted = pd.melt(totaldata, id_vars=['ClassType'], value_vars=['Prob02', 'Prob04', 'Prob06'],
                          var_name='Probs', value_name='Amount')
totaldata_melted

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ClassType</th>
      <th>Probs</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sheep</td>
      <td>Prob02</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bottle</td>
      <td>Prob02</td>
      <td>42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>horse</td>
      <td>Prob02</td>
      <td>101</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bicycle</td>
      <td>Prob02</td>
      <td>168</td>
    </tr>
    <tr>
      <th>4</th>
      <td>motorbike</td>
      <td>Prob02</td>
      <td>50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cow</td>
      <td>Prob02</td>
      <td>17</td>
    </tr>
    <tr>
      <th>6</th>
      <td>bus</td>
      <td>Prob02</td>
      <td>21</td>
    </tr>
    <tr>
      <th>7</th>
      <td>dog</td>
      <td>Prob02</td>
      <td>74</td>
    </tr>
    <tr>
      <th>8</th>
      <td>cat</td>
      <td>Prob02</td>
      <td>4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>person</td>
      <td>Prob02</td>
      <td>4688</td>
    </tr>
    <tr>
      <th>10</th>
      <td>train</td>
      <td>Prob02</td>
      <td>41</td>
    </tr>
    <tr>
      <th>11</th>
      <td>diningtable</td>
      <td>Prob02</td>
      <td>70</td>
    </tr>
    <tr>
      <th>12</th>
      <td>aeroplane</td>
      <td>Prob02</td>
      <td>32</td>
    </tr>
    <tr>
      <th>13</th>
      <td>sofa</td>
      <td>Prob02</td>
      <td>54</td>
    </tr>
    <tr>
      <th>14</th>
      <td>pottedplant</td>
      <td>Prob02</td>
      <td>146</td>
    </tr>
    <tr>
      <th>15</th>
      <td>tvmonitor</td>
      <td>Prob02</td>
      <td>85</td>
    </tr>
    <tr>
      <th>16</th>
      <td>chair</td>
      <td>Prob02</td>
      <td>357</td>
    </tr>
    <tr>
      <th>17</th>
      <td>bird</td>
      <td>Prob02</td>
      <td>42</td>
    </tr>
    <tr>
      <th>18</th>
      <td>boat</td>
      <td>Prob02</td>
      <td>102</td>
    </tr>
    <tr>
      <th>19</th>
      <td>car</td>
      <td>Prob02</td>
      <td>192</td>
    </tr>
    <tr>
      <th>20</th>
      <td>sheep</td>
      <td>Prob04</td>
      <td>7</td>
    </tr>
    <tr>
      <th>21</th>
      <td>bottle</td>
      <td>Prob04</td>
      <td>9</td>
    </tr>
    <tr>
      <th>22</th>
      <td>horse</td>
      <td>Prob04</td>
      <td>76</td>
    </tr>
    <tr>
      <th>23</th>
      <td>bicycle</td>
      <td>Prob04</td>
      <td>92</td>
    </tr>
    <tr>
      <th>24</th>
      <td>motorbike</td>
      <td>Prob04</td>
      <td>25</td>
    </tr>
    <tr>
      <th>25</th>
      <td>cow</td>
      <td>Prob04</td>
      <td>5</td>
    </tr>
    <tr>
      <th>26</th>
      <td>bus</td>
      <td>Prob04</td>
      <td>13</td>
    </tr>
    <tr>
      <th>27</th>
      <td>dog</td>
      <td>Prob04</td>
      <td>46</td>
    </tr>
    <tr>
      <th>28</th>
      <td>cat</td>
      <td>Prob04</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>person</td>
      <td>Prob04</td>
      <td>3367</td>
    </tr>
    <tr>
      <th>30</th>
      <td>train</td>
      <td>Prob04</td>
      <td>21</td>
    </tr>
    <tr>
      <th>31</th>
      <td>diningtable</td>
      <td>Prob04</td>
      <td>41</td>
    </tr>
    <tr>
      <th>32</th>
      <td>aeroplane</td>
      <td>Prob04</td>
      <td>18</td>
    </tr>
    <tr>
      <th>33</th>
      <td>sofa</td>
      <td>Prob04</td>
      <td>39</td>
    </tr>
    <tr>
      <th>34</th>
      <td>pottedplant</td>
      <td>Prob04</td>
      <td>73</td>
    </tr>
    <tr>
      <th>35</th>
      <td>tvmonitor</td>
      <td>Prob04</td>
      <td>48</td>
    </tr>
    <tr>
      <th>36</th>
      <td>chair</td>
      <td>Prob04</td>
      <td>186</td>
    </tr>
    <tr>
      <th>37</th>
      <td>bird</td>
      <td>Prob04</td>
      <td>22</td>
    </tr>
    <tr>
      <th>38</th>
      <td>boat</td>
      <td>Prob04</td>
      <td>65</td>
    </tr>
    <tr>
      <th>39</th>
      <td>car</td>
      <td>Prob04</td>
      <td>119</td>
    </tr>
    <tr>
      <th>40</th>
      <td>sheep</td>
      <td>Prob06</td>
      <td>4</td>
    </tr>
    <tr>
      <th>41</th>
      <td>bottle</td>
      <td>Prob06</td>
      <td>3</td>
    </tr>
    <tr>
      <th>42</th>
      <td>horse</td>
      <td>Prob06</td>
      <td>64</td>
    </tr>
    <tr>
      <th>43</th>
      <td>bicycle</td>
      <td>Prob06</td>
      <td>61</td>
    </tr>
    <tr>
      <th>44</th>
      <td>motorbike</td>
      <td>Prob06</td>
      <td>14</td>
    </tr>
    <tr>
      <th>45</th>
      <td>cow</td>
      <td>Prob06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46</th>
      <td>bus</td>
      <td>Prob06</td>
      <td>11</td>
    </tr>
    <tr>
      <th>47</th>
      <td>dog</td>
      <td>Prob06</td>
      <td>29</td>
    </tr>
    <tr>
      <th>48</th>
      <td>cat</td>
      <td>Prob06</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>person</td>
      <td>Prob06</td>
      <td>2699</td>
    </tr>
    <tr>
      <th>50</th>
      <td>train</td>
      <td>Prob06</td>
      <td>11</td>
    </tr>
    <tr>
      <th>51</th>
      <td>diningtable</td>
      <td>Prob06</td>
      <td>26</td>
    </tr>
    <tr>
      <th>52</th>
      <td>aeroplane</td>
      <td>Prob06</td>
      <td>13</td>
    </tr>
    <tr>
      <th>53</th>
      <td>sofa</td>
      <td>Prob06</td>
      <td>24</td>
    </tr>
    <tr>
      <th>54</th>
      <td>pottedplant</td>
      <td>Prob06</td>
      <td>45</td>
    </tr>
    <tr>
      <th>55</th>
      <td>tvmonitor</td>
      <td>Prob06</td>
      <td>36</td>
    </tr>
    <tr>
      <th>56</th>
      <td>chair</td>
      <td>Prob06</td>
      <td>112</td>
    </tr>
    <tr>
      <th>57</th>
      <td>bird</td>
      <td>Prob06</td>
      <td>14</td>
    </tr>
    <tr>
      <th>58</th>
      <td>boat</td>
      <td>Prob06</td>
      <td>45</td>
    </tr>
    <tr>
      <th>59</th>
      <td>car</td>
      <td>Prob06</td>
      <td>89</td>
    </tr>
  </tbody>
</table>
</div>




```python
totaldata_melted.to_csv("/home/veronika/materials/cv/detection/total_classes.csv")
```


```python
All details about SSD are here: https://github.com/weiliu89/caffe/tree/ssd/examples
```
