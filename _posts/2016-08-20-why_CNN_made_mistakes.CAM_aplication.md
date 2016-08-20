---
layout: post
title: Why my model made mistakes. Application of CAM. Part 2
subtitle: by Veronika Yurchuk
bigimg: /img/img13.jpg
show-avatar: false
---

I was very impressed after analysing how Convolutional Neural Network see the image. More about it in the [link](https://veronikayurchuk.github.io/2016-08-14-The-way-how-ConvNet-see-the-images-Application-of-CAM/).

As any model in Machine Learning makes some mistake, I would like to try to analyze why my model made that mistakes. In order to solve this question we can use CAM and compare results. Moreover, trying to understand why model made some mistakes we have to look at confusion matrix. Confusion matrix shows us which classes were missclassified the most.

So we apload confusion matrix, that we have already seen on this [post](https://veronikayurchuk.github.io/2016-08-14-more_about_CNN_training_and_transflearn/).


```python
%matplotlib inline
import matplotlib.image as mpimg
plt.rcParams['figure.figsize'] = (22, 9)
confusion_m = mpimg.imread('/home/veronika/materials/cv/cv_organizer/presentation/confmatrix.png')
plt.imshow(confusion_m)
```




    <matplotlib.image.AxesImage at 0x7f66ba5b8650>




![png](/img/post6/output_1_1.png)



```python
class_label = pd.read_csv("/home/veronika/materials/cv/detection/class_labels.csv", sep = " ", header = None)
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



After analysing confusion matrix we can make the following conclusions. The most repetitive wrong predicted case was when true lable of image is "runninig/walking" and predicted class is "sports" There were 24 errors of this type. However, that cases are very similar. Other high repetitive cases are
- when true class is "home_repair", but predicted class is "ocupation"
- when true class is "home_activities", but predicted class is "occupation"
- when true class is "other", but predicted class is "sports"
- when true class is "occupation", but predicted class is "other"

OK! Let's look on features, that mislead our model. This analysis may help us to understand the reasons of misclassifications. It might be bad labeled images, that are difficult to classify even for human or some objects in images, that led to errors.


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
    nclass = 13
    sortedprobs = np.sort(detections)
    
    
    total_data.set_value(true_test_labels.loc[true_test_labels["name"] == i].index[0],
                         'predicted_label', np.argsort(detections)[:,nclass])
    
    total_data.set_value(true_test_labels.loc[true_test_labels["name"] == i].index[0],
                         'prob', sortedprobs[:,nclass])
    total_data.set_value(true_test_labels.loc[true_test_labels["name"] == i].index[0],
                         'second_predicted_label', np.argsort(detections)[:,nclass-1])
    total_data.set_value(true_test_labels.loc[true_test_labels["name"] == i].index[0],
                         'second_prob', sortedprobs[:,nclass-1])
    
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
total_data.shape
```




    (1798, 6)




```python
total_data.tail(7)
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
      <th>1791</th>
      <td>056481278.jpg</td>
      <td>11</td>
      <td>0</td>
      <td>0.738038</td>
      <td>1</td>
      <td>0.210865</td>
    </tr>
    <tr>
      <th>1792</th>
      <td>001135364.jpg</td>
      <td>13</td>
      <td>13</td>
      <td>0.683123</td>
      <td>9</td>
      <td>0.070605</td>
    </tr>
    <tr>
      <th>1793</th>
      <td>015683370.jpg</td>
      <td>10</td>
      <td>11</td>
      <td>0.942924</td>
      <td>9</td>
      <td>0.038226</td>
    </tr>
    <tr>
      <th>1794</th>
      <td>001367765.jpg</td>
      <td>12</td>
      <td>12</td>
      <td>0.999849</td>
      <td>7</td>
      <td>0.000055</td>
    </tr>
    <tr>
      <th>1795</th>
      <td>090711742.jpg</td>
      <td>2</td>
      <td>2</td>
      <td>0.968152</td>
      <td>1</td>
      <td>0.031011</td>
    </tr>
    <tr>
      <th>1796</th>
      <td>043042191.jpg</td>
      <td>5</td>
      <td>0</td>
      <td>0.470264</td>
      <td>5</td>
      <td>0.240976</td>
    </tr>
    <tr>
      <th>1797</th>
      <td>094023024.jpg</td>
      <td>12</td>
      <td>12</td>
      <td>0.998844</td>
      <td>4</td>
      <td>0.000477</td>
    </tr>
  </tbody>
</table>
</div>




```python
wrong_predicted = total_data[total_data["true_label"] != total_data["predicted_label"]]
wrong_predicted.tail(7)

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
      <th>1777</th>
      <td>000391837.jpg</td>
      <td>10</td>
      <td>0</td>
      <td>0.672848</td>
      <td>1</td>
      <td>0.092628</td>
    </tr>
    <tr>
      <th>1778</th>
      <td>016859905.jpg</td>
      <td>6</td>
      <td>8</td>
      <td>0.267946</td>
      <td>6</td>
      <td>0.266094</td>
    </tr>
    <tr>
      <th>1781</th>
      <td>000947513.jpg</td>
      <td>8</td>
      <td>11</td>
      <td>0.489302</td>
      <td>1</td>
      <td>0.169710</td>
    </tr>
    <tr>
      <th>1787</th>
      <td>023722679.jpg</td>
      <td>7</td>
      <td>10</td>
      <td>0.668150</td>
      <td>7</td>
      <td>0.210047</td>
    </tr>
    <tr>
      <th>1791</th>
      <td>056481278.jpg</td>
      <td>11</td>
      <td>0</td>
      <td>0.738038</td>
      <td>1</td>
      <td>0.210865</td>
    </tr>
    <tr>
      <th>1793</th>
      <td>015683370.jpg</td>
      <td>10</td>
      <td>11</td>
      <td>0.942924</td>
      <td>9</td>
      <td>0.038226</td>
    </tr>
    <tr>
      <th>1796</th>
      <td>043042191.jpg</td>
      <td>5</td>
      <td>0</td>
      <td>0.470264</td>
      <td>5</td>
      <td>0.240976</td>
    </tr>
  </tbody>
</table>
</div>




```python
wrong_predicted.shape
```




    _(431, 6)_




```python
wrong_predicted['nindex'] = range(431)
```

    /usr/local/lib/python2.7/dist-packages/ipykernel/__main__.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    

```python
wrong_predicted.head(7)
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
      <th>nindex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>089393351.jpg</td>
      <td>6</td>
      <td>7</td>
      <td>0.818566</td>
      <td>6</td>
      <td>0.125225</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>064173704.jpg</td>
      <td>1</td>
      <td>11</td>
      <td>0.469895</td>
      <td>0</td>
      <td>0.300765</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>033109429.jpg</td>
      <td>7</td>
      <td>10</td>
      <td>0.462610</td>
      <td>7</td>
      <td>0.426548</td>
      <td>2</td>
    </tr>
    <tr>
      <th>28</th>
      <td>033281836.jpg</td>
      <td>13</td>
      <td>11</td>
      <td>0.467383</td>
      <td>13</td>
      <td>0.288317</td>
      <td>3</td>
    </tr>
    <tr>
      <th>37</th>
      <td>048507291.jpg</td>
      <td>10</td>
      <td>8</td>
      <td>0.342671</td>
      <td>1</td>
      <td>0.255119</td>
      <td>4</td>
    </tr>
    <tr>
      <th>39</th>
      <td>060381722.jpg</td>
      <td>10</td>
      <td>7</td>
      <td>0.482599</td>
      <td>6</td>
      <td>0.298314</td>
      <td>5</td>
    </tr>
    <tr>
      <th>52</th>
      <td>067799802.jpg</td>
      <td>7</td>
      <td>10</td>
      <td>0.790746</td>
      <td>7</td>
      <td>0.097995</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("There are" + " " + str(wrong_predicted.shape[0])+" " + "wrong predicted images")
```

    _There are 431 wrong predicted images_



```python
def get_label(n, label_type):
    return(np.array(class_label.loc[class_label[0] == wrong_predicted.iloc[n][label_type]][1])[0])
```


```python
def get_heat_map(n, argtype = "index"):
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
    snd_pred_label = "Pred label" +":"+ " " + get_label(i, "second_predicted_label") +" "+ str(round(wrong_predicted.iloc[i]["second_prob"], 3))
    plt.text(25, 0.04*image.shape[1], tr_label, {'color': 'w', 'fontsize': 21})
    plt.text(25, 2*0.04*image.shape[1], pred_label, {'color': 'w', 'fontsize': 21})
    plt.text(25, 3*0.04*image.shape[1], snd_pred_label, {'color': 'w', 'fontsize': 21})
    plt.imshow(get_heat_map(i), alpha=0.4, interpolation='nearest')
    plt.imshow(image)
    return(plt.show())
```

Now, we will look at most repetitive error cases (true class of image is "runninig/walking" and predicted class is "sports")



```python
wrong_predicted.loc[wrong_predicted["true_label"] == 0][wrong_predicted["predicted_label"] == 11].head(12)
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
      <th>nindex</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>196</th>
      <td>008058081.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.816078</td>
      <td>0</td>
      <td>0.165102</td>
      <td>40</td>
    </tr>
    <tr>
      <th>217</th>
      <td>091082626.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.611820</td>
      <td>0</td>
      <td>0.208546</td>
      <td>45</td>
    </tr>
    <tr>
      <th>224</th>
      <td>012324746.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.857252</td>
      <td>3</td>
      <td>0.092970</td>
      <td>46</td>
    </tr>
    <tr>
      <th>390</th>
      <td>069342584.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.483682</td>
      <td>0</td>
      <td>0.372490</td>
      <td>90</td>
    </tr>
    <tr>
      <th>393</th>
      <td>030264034.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.988194</td>
      <td>0</td>
      <td>0.008093</td>
      <td>91</td>
    </tr>
    <tr>
      <th>450</th>
      <td>057277028.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.783435</td>
      <td>0</td>
      <td>0.151723</td>
      <td>106</td>
    </tr>
    <tr>
      <th>482</th>
      <td>025283378.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.805813</td>
      <td>0</td>
      <td>0.166451</td>
      <td>114</td>
    </tr>
    <tr>
      <th>676</th>
      <td>074163782.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.904602</td>
      <td>0</td>
      <td>0.073850</td>
      <td>164</td>
    </tr>
    <tr>
      <th>728</th>
      <td>005773628.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.450017</td>
      <td>6</td>
      <td>0.288198</td>
      <td>180</td>
    </tr>
    <tr>
      <th>787</th>
      <td>085793065.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.634702</td>
      <td>13</td>
      <td>0.205134</td>
      <td>195</td>
    </tr>
    <tr>
      <th>879</th>
      <td>095797935.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.880209</td>
      <td>0</td>
      <td>0.091266</td>
      <td>221</td>
    </tr>
    <tr>
      <th>904</th>
      <td>004997903.jpg</td>
      <td>0</td>
      <td>11</td>
      <td>0.548356</td>
      <td>0</td>
      <td>0.420405</td>
      <td>226</td>
    </tr>
  </tbody>
</table>
</div>



In the picture bellow we can see that ConvNet has fixed correctly on a person, but doesn't manage to classify it to correct label. Also, the correct label was defined with 16.5%. 


```python
n = 40
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_27_0.png)





    <matplotlib.image.AxesImage at 0x7f66b9e2d290>




![png](/img/post6/output_27_2.png)


In his case, the ConvNet didn't detect the person.


```python
n = 45
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_29_0.png)





    <matplotlib.image.AxesImage at 0x7f66b965b550>




![png](/img/post6/output_29_2.png)


Let's look at the next image. I think that it is not easy to detect the running activities at the image,as we can't see the whole person. Nevertheless, the probability of correct class is high.


```python
n = 90
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_31_0.png)





    <matplotlib.image.AxesImage at 0x7f66b9f59f10>




![png](/img/post6/output_31_2.png)


The same situation as above


```python
n = 114
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_33_0.png)





    <matplotlib.image.AxesImage at 0x7f66b95adc90>




![png](/img/post6/output_33_2.png)


As we can see, ***not only the model made mistakes, but human too.*** As there are a lot of cases of not very clear labeling. You may check it looking at images bellow.


```python
n = 4
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_35_0.png)





    <matplotlib.image.AxesImage at 0x7f66c0c13b10>




![png](/img/post6/output_35_2.png)



```python
n = 7
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_36_0.png)





    <matplotlib.image.AxesImage at 0x7f66c0043e10>




![png](/img/post6/output_36_2.png)



```python
n = 10
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_37_0.png)





    <matplotlib.image.AxesImage at 0x7f66c16a8d50>




![png](/img/post6/output_37_2.png)



```python
n = 11
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_38_0.png)





    <matplotlib.image.AxesImage at 0x7f66bbf2bbd0>




![png](/img/post6/output_38_2.png)



```python
n = 12
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_39_0.png)





    <matplotlib.image.AxesImage at 0x7f66bbbe7790>




![png](/img/post6/output_39_2.png)



```python
n = 14
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_40_0.png)





    <matplotlib.image.AxesImage at 0x7f66bbb53350>




![png]/img/post6/output_40_2.png)



```python
n = 15
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_41_0.png)





    <matplotlib.image.AxesImage at 0x7f66bbd6da90>




![png](/img/post6/output_41_2.png)



```python
n = 16
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_42_0.png)





    <matplotlib.image.AxesImage at 0x7f66c0d71690>




![png](/img/post6/output_42_2.png)



```python
n = 19
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_43_0.png)





    <matplotlib.image.AxesImage at 0x7f66bb7b9610>




![png](/img/post6/output_43_2.png)



```python
n = 20
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_44_0.png)





    <matplotlib.image.AxesImage at 0x7f66bb9201d0>




![png](/img/post6/output_44_2.png)



```python
n = 35
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_45_0.png)





    <matplotlib.image.AxesImage at 0x7f66bbe28f50>




![png](/img/post6/output_45_2.png)



```python
n = 36
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_46_0.png)





    <matplotlib.image.AxesImage at 0x7f66bb7703d0>




![png]/img/post6/(output_46_2.png)



```python
n = 60
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_47_0.png)





    <matplotlib.image.AxesImage at 0x7f66bb11bdd0>




![png](/img/post6/output_47_2.png)



```python
n = 68
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_48_0.png)





    <matplotlib.image.AxesImage at 0x7f66baf53990>




![png](/img/post6/output_48_2.png)



```python
n = 69
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_49_0.png)





    <matplotlib.image.AxesImage at 0x7f66badc1550>




![png](/img/post6/output_49_2.png)



```python
n = 75
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_50_0.png)





    <matplotlib.image.AxesImage at 0x7f66bb0a5910>




![png](/img/post6/output_50_2.png)



```python
n = 78
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_51_0.png)





    <matplotlib.image.AxesImage at 0x7f66bb2aa2d0>




![png](/img/post6/output_51_2.png)



```python
n = 87
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_52_0.png)





    <matplotlib.image.AxesImage at 0x7f66bbe20890>




![png](/img/post6/output_52_2.png)



```python
n = 88
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_53_0.png)





    <matplotlib.image.AxesImage at 0x7f66bb1c25d0>




![png](/img/post6/output_53_2.png)



```python
n = 95
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_54_0.png)





    <matplotlib.image.AxesImage at 0x7f66baee1190>




![png](/img/post6/output_54_2.png)



```python
n = 99
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_55_0.png)





    <matplotlib.image.AxesImage at 0x7f66baac5d10>




![png](/img/post6/output_55_2.png)



```python
n = 103
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_56_0.png)





    <matplotlib.image.AxesImage at 0x7f66babaa8d0>




![png](/img/post6/output_56_2.png)



```python
n = 104
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_57_0.png)





    <matplotlib.image.AxesImage at 0x7f66baaa1410>




![png](/img/post6/output_57_2.png)



```python
n = 110
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_58_0.png)





    <matplotlib.image.AxesImage at 0x7f66bb0449d0>




![png](/img/post6/output_58_2.png)



```python
n = 118
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_59_0.png)





    <matplotlib.image.AxesImage at 0x7f66bb7c8a90>




![png](/img/post6/output_59_2.png)



```python
n = 121
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_60_0.png)





    <matplotlib.image.AxesImage at 0x7f66baf1e950>




![png](/img/post6/output_60_2.png)



```python
n = 122
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_61_0.png)





    <matplotlib.image.AxesImage at 0x7f66ba8db510>




![png](/img/post6/output_61_2.png)



```python
n = 9
plot_labeled_image(n)
plt.imshow(get_heat_map(n), alpha=0.4, interpolation='nearest')
```


![png](/img/post6/output_62_0.png)





    <matplotlib.image.AxesImage at 0x7f66ba6a6c50>




![png](/img/post6/output_62_2.png)

