
I was very impressed after analysing how Convolutional Neural Network see the image. More about it in the [link](https://veronikayurchuk.github.io/2016-08-14-The-way-how-ConvNet-see-the-images-Application-of-CAM/).

Any model in Machine Learning makes some mistake. In this post I will try to analyze why my model made that mistakes. In order to solve this question we can use CAM and compare results. Moreover, trying to understand why model made some mistakes we have to look at confusion matrix. Confusion matrix shows us which classes were missclassified the most.


```python
import caffe
import numpy as np
import cv2
import matplotlib.pylab as plt
import sys
import pandas as pd
```

    /usr/local/lib/python2.7/dist-packages/matplotlib/font_manager.py:273: UserWarning: Matplotlib is building the font cache using fc-list. This may take a moment.
      warnings.warn('Matplotlib is building the font cache using fc-list. This may take a moment.')



```python
%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 9)
```


```python
deploy_file  = "/home/veronika/materials/cv/CAM/deploy_googlenetCAM.prototxt"
weights_file = "/home/veronika/materials/cv/CAM/mod2/snapshot/_iter_3000.caffemodel"

caffe.set_device(0)
caffe.set_mode_gpu()
```


```python
net = caffe.Net(deploy_file, weights_file, caffe.TEST)
```


```python
class_label = pd.read_csv("/home/veronika/materials/cv/detection/class_labels.csv", sep = " ", header = None)

```


```python
true_test_labels = pd.read_csv("/home/veronika/materials/cv/cv_organizer/mydata/test_labels.csv", 
                               sep = " ", names=["name", "true_label"])
```


```python
total_data = true_test_labels
total_data.shape
```




    (1798, 2)




```python
for i in true_test_labels['name']:
    img_path = "/home/veronika/materials/cv/cv_organizer/mydata/test/" + i
    image_resize = 224
    image = caffe.io.load_image(img_path)
    net.blobs['data'].reshape(1,3,image_resize,image_resize)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    detections = net.forward()['prob']
    total_data.set_value(true_test_labels.loc[true_test_labels["name"] == i].index[0],
                         'predicted_label', detections.argmax())
    total_data.set_value(true_test_labels.loc[true_test_labels["name"] == i].index[0],
                         'prob', detections.max())
    total_data.set_value(true_test_labels.loc[true_test_labels["name"] == i].index[0],
                         'second_predicted_label', detections[np.where(detections != detections.max())].argmax()+1)
    total_data.set_value(true_test_labels.loc[true_test_labels["name"] == i].index[0],
                         'second_prob', detections[np.where(detections != detections.max())].max())
    
```


```python
total_data["predicted_label"] = (total_data["predicted_label"]).astype(int)
total_data["second_predicted_label"] = (total_data["second_predicted_label"]).astype(int)
total_data["true_label"] = (total_data["true_label"]).astype(int)
```


```python
total_data = pd.read_csv("/home/veronika/materials/cv/predicted_labels_CAM.csv")
```


```python
sum(total_data["predicted_label"] == total_data["second_predicted_label"])
```




    141




```python
total_data.shape
```




    (1798, 6)




```python
total_data.head(7)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>true_label</th>
      <th>predicted_label</th>
      <th>prob</th>
      <th>second_predicted_label</th>
      <th>second_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>010814960.jpg</td>
      <td>1</td>
      <td>1</td>
      <td>0.976134</td>
      <td>6</td>
      <td>0.012126</td>
    </tr>
    <tr>
      <th>1</th>
      <td>027871210.jpg</td>
      <td>11</td>
      <td>11</td>
      <td>0.990322</td>
      <td>4</td>
      <td>0.009666</td>
    </tr>
    <tr>
      <th>2</th>
      <td>089393351.jpg</td>
      <td>6</td>
      <td>7</td>
      <td>0.818566</td>
      <td>7</td>
      <td>0.125225</td>
    </tr>
    <tr>
      <th>3</th>
      <td>019668643.jpg</td>
      <td>11</td>
      <td>11</td>
      <td>0.822453</td>
      <td>2</td>
      <td>0.094344</td>
    </tr>
    <tr>
      <th>4</th>
      <td>075237453.jpg</td>
      <td>3</td>
      <td>3</td>
      <td>0.960489</td>
      <td>4</td>
      <td>0.022237</td>
    </tr>
    <tr>
      <th>5</th>
      <td>037504070.jpg</td>
      <td>2</td>
      <td>2</td>
      <td>0.993954</td>
      <td>2</td>
      <td>0.005957</td>
    </tr>
    <tr>
      <th>6</th>
      <td>064173704.jpg</td>
      <td>1</td>
      <td>11</td>
      <td>0.469895</td>
      <td>1</td>
      <td>0.300765</td>
    </tr>
  </tbody>
</table>
</div>




```python
wrong_predicted = total_data[total_data["true_label"] != total_data["predicted_label"]]
wrong_predicted.head()

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>true_label</th>
      <th>predicted_label</th>
      <th>prob</th>
      <th>second_predicted_label</th>
      <th>second_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>089393351.jpg</td>
      <td>6</td>
      <td>7</td>
      <td>0.818566</td>
      <td>7</td>
      <td>0.125225</td>
    </tr>
    <tr>
      <th>6</th>
      <td>064173704.jpg</td>
      <td>1</td>
      <td>11</td>
      <td>0.469895</td>
      <td>1</td>
      <td>0.300765</td>
    </tr>
    <tr>
      <th>9</th>
      <td>033109429.jpg</td>
      <td>7</td>
      <td>10</td>
      <td>0.462610</td>
      <td>8</td>
      <td>0.426548</td>
    </tr>
    <tr>
      <th>28</th>
      <td>033281836.jpg</td>
      <td>13</td>
      <td>11</td>
      <td>0.467383</td>
      <td>13</td>
      <td>0.288317</td>
    </tr>
    <tr>
      <th>37</th>
      <td>048507291.jpg</td>
      <td>10</td>
      <td>8</td>
      <td>0.342671</td>
      <td>2</td>
      <td>0.255119</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("There are" + " " + str(wrong_predicted.shape[0])+" " + "wrong predicted images")
```

    There are 431 wrong predicted images



```python
class_label
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>running/walking</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>other</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>bicycling</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>conditioning_exercise</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>dancing</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>fishing_and_hunting</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>home_activities</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>home_repair</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>lawn_and_garden</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>music_playing</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>occupation</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>sports</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>water_activities</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>winter_activities</td>
    </tr>
  </tbody>
</table>
</div>




```python
def get_label(n, label_type):
    return(np.array(class_label.loc[class_label[0] == wrong_predicted.iloc[n][label_type]][1])[0])
```


```python
def get_heat_map(n):
    image_resize = 224
    img_path = "/home/veronika/materials/cv/cv_organizer/mydata/test/" + np.array(wrong_predicted["name"])[n]
    image = caffe.io.load_image(img_path)
    net.blobs['data'].reshape(1,3,image_resize,image_resize)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104,117,123])) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    detections = net.forward()['prob']
    weights = net.params['CAM_fc_14'][0].data
    feature_maps = net.blobs['CAM_conv'].data[0]

    weights = np.array(weights,dtype=np.float)
    feature_maps = np.array(feature_maps,dtype=np.float)
    heat_map = np.zeros([14,14],dtype = np.float)
    for i in range(1024):
        w = weights[detections.argmax()][i]
        heat_map = heat_map + w*feature_maps[i]
    heat_map = cv2.resize(heat_map,(224,224))
    return(heat_map)
```


```python
def plot_labeled_image(i):
    image_resize = 224
    img_path = "/home/veronika/materials/cv/cv_organizer/mydata/test/" + np.array(wrong_predicted["name"])[i]
    image = caffe.io.load_image(img_path)
    image = cv2.resize(image,(224,224))
    tr_label = "True label" +":"+ " " + get_label(i, "true_label")
    pred_label = "Pred label" +":"+ " " + get_label(i, "predicted_label") +" "+ str(round(wrong_predicted.iloc[i]["prob"], 3))
    #snd_pred_label = "Pred label" +":"+ " " + get_label(i, "second_predicted_label") +" "+ str(round(wrong_predicted.iloc[i]["second_prob"], 3))
    plt.text(25, 0.04*image.shape[1], tr_label, {'color': 'w', 'fontsize': 21})
    plt.text(25, 2*0.04*image.shape[1], pred_label, {'color': 'w', 'fontsize': 22})
    #plt.text(50, 150, snd_pred_label, {'color': 'w', 'fontsize': 30})
    plt.imshow(get_heat_map(i), alpha=0.4, interpolation='nearest')
    plt.imshow(image)
    return(plt.show())
```


```python
get_heat_map(5)
```




    array([[ 0.77282175,  0.77282175,  0.77282175, ...,  3.94781259,
             3.94781259,  3.94781259],
           [ 0.77282175,  0.77282175,  0.77282175, ...,  3.94781259,
             3.94781259,  3.94781259],
           [ 0.77282175,  0.77282175,  0.77282175, ...,  3.94781259,
             3.94781259,  3.94781259],
           ..., 
           [ 6.85190451,  6.85190451,  6.85190451, ..., -5.26715516,
            -5.26715516, -5.26715516],
           [ 6.85190451,  6.85190451,  6.85190451, ..., -5.26715516,
            -5.26715516, -5.26715516],
           [ 6.85190451,  6.85190451,  6.85190451, ..., -5.26715516,
            -5.26715516, -5.26715516]])




```python
n = 1
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_21_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf8471190>




![png](output_21_2.png)



```python
n = 4
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_22_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf16d0a10>




![png](output_22_2.png)



```python
n = 5
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_23_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf1704190>




![png](output_23_2.png)



```python
n = 6
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_24_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf15ddf10>




![png](output_24_2.png)



```python
n = 7
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_25_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf85ab950>




![png](output_25_2.png)



```python
n = 8
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_26_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf8420a90>




![png](output_26_2.png)



```python
n = 9
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_27_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf1440150>




![png](output_27_2.png)



```python
n = 10
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_28_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf136a890>




![png](output_28_2.png)



```python
n = 11
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_29_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf1384fd0>




![png](output_29_2.png)



```python
n = 12
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_30_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf1761cd0>




![png](output_30_2.png)



```python
n = 14
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_31_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf1222a90>




![png](output_31_2.png)



```python
n = 15
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_32_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf0f12210>




![png](output_32_2.png)



```python
n = 16
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_33_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf0d77950>




![png](output_33_2.png)



```python
n = 17
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_34_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf0c200d0>




![png](output_34_2.png)



```python
n = 18
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_35_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf1121a50>




![png](output_35_2.png)



```python
n = 19
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_36_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf17e0110>




![png](output_36_2.png)



```python
n = 20
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_37_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf1193910>




![png](output_37_2.png)



```python
n = 24
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_38_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf0b14c50>




![png](output_38_2.png)



```python
n = 35
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_39_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf01cd790>




![png](output_39_2.png)



```python
n = 36
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_40_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf052eed0>




![png](output_40_2.png)



```python
n = 37
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_41_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf00d8650>




![png](output_41_2.png)



```python
n = 43
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_42_0.png)





    <matplotlib.image.AxesImage at 0x7fcaf09250d0>




![png](output_42_2.png)



```python
n = 45
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_43_0.png)





    <matplotlib.image.AxesImage at 0x7fcaeba04c90>




![png](output_43_2.png)



```python
n = 46
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_44_0.png)





    <matplotlib.image.AxesImage at 0x7fcaeb936410>




![png](output_44_2.png)



```python
n = 55
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_45_0.png)





    <matplotlib.image.AxesImage at 0x7fcaeb2ca1d0>




![png](output_45_2.png)



```python
n = 58
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_46_0.png)





    <matplotlib.image.AxesImage at 0x7fcaeae6d7d0>




![png](output_46_2.png)



```python
n = 60
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_47_0.png)





    <matplotlib.image.AxesImage at 0x7fcaeb40c750>




![png](output_47_2.png)



```python
n = 68
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_48_0.png)





    <matplotlib.image.AxesImage at 0x7fcaea7efa90>




![png](output_48_2.png)



```python
n = 69
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_49_0.png)





    <matplotlib.image.AxesImage at 0x7fcaea8da210>




![png](output_49_2.png)



```python
n = 75
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_50_0.png)





    <matplotlib.image.AxesImage at 0x7fcaea95e510>




![png](output_50_2.png)



```python
n = 78
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_51_0.png)





    <matplotlib.image.AxesImage at 0x7fcaea39eb10>




![png](output_51_2.png)



```python
n = 87
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_52_0.png)





    <matplotlib.image.AxesImage at 0x7fcaea311950>




![png](output_52_2.png)



```python
n = 88
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_53_0.png)





    <matplotlib.image.AxesImage at 0x7fcae9ce30d0>




![png](output_53_2.png)



```python
n = 95
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_54_0.png)





    <matplotlib.image.AxesImage at 0x7fcae932b510>




![png](output_54_2.png)



```python
n = 99
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_55_0.png)





    <matplotlib.image.AxesImage at 0x7fcaeb96c290>




![png](output_55_2.png)



```python
n = 103
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_56_0.png)





    <matplotlib.image.AxesImage at 0x7fcae9266fd0>




![png](output_56_2.png)



```python
n = 104
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_57_0.png)





    <matplotlib.image.AxesImage at 0x7fcae908e750>




![png](output_57_2.png)



```python
n = 110
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_58_0.png)





    <matplotlib.image.AxesImage at 0x7fcae8670390>




![png](output_58_2.png)



```python
n = 118
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_59_0.png)





    <matplotlib.image.AxesImage at 0x7fcae95ba850>




![png](output_59_2.png)



```python
n = 121
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_60_0.png)





    <matplotlib.image.AxesImage at 0x7fcae84f7fd0>




![png](output_60_2.png)



```python
n = 122
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_61_0.png)





    <matplotlib.image.AxesImage at 0x7fcae8320750>




![png](output_61_2.png)



```python
n = 132
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](output_62_0.png)





    <matplotlib.image.AxesImage at 0x7fcae93fa390>




![png](output_62_2.png)



```python
n = 9
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```
