---
layout: post
title: More about CNN training and transfer learning
subtitle: by Veronika Yurchuk
bigimg: /img/path.jpg
show-avatar: false
---


# Analysis of results and playing with featuremaps  

### In order to train model for activity classification I used pre-trained model, trained on ImageNet. The next step was to apply transfer learning. What is it? The main idea is to use weights that were trained on ImageNet and change them a bit to let them learn from other dataset. Important to mention that we have to change learning rate too. It should be smaller for layers that are the same with original model and be heiger for new layers. Next step is fine-tuning hyperparameters.

### After fine-tuning various models and trying lots of learning rate policies, we got our best result with ***accuracy = 95%*** on test set, that is 10% of all MPII dataset. I have tried Adam, RMSProp and Momentum stochastic gradient optimization algorithms.
### The best optimization algorithm was ***Adam***, the best topology was GoogLeNet and my learning rate policy:
base_lr: 0.0005
lr_policy: "step"
stepsize: 2000

### All files for trainig models and my pre-trained weights you may find in the github repo.

You may find the plot of learning curves below:


```python
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.rcParams['figure.figsize'] = (18.0, 10.0)

learning_curves = mpimg.imread('/home/veronika/materials/cv/cv_organizer/googlenet/adam4/best.png')
plt.imshow(learning_curves)
```




    <matplotlib.image.AxesImage at 0x7fc115efca90>




![png](/img/post4/output_3_1.png)


Using the following function we can get features from defined layer for each features.


```python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import caffe
import cv2

def probs_feats(img_path, deploy_path, weights_path, blob_name="pool5/7x7_s1"):
    channels = 3
    rows = 224
    cols = 224

    net = caffe.Net(deploy_path, weights_path, caffe.TEST)
    net.blobs['data'].reshape(1, channels, rows, cols)

    image = caffe.io.load_image(img_path)
    image = cv2.resize(image, (224, 224))
    image = image.swapaxes(0,2).swapaxes(1,2)
    image = image.reshape(1, channels, rows, cols)
    input_img = image.astype(float)
    net.blobs["data"].data[...] = input_img
    probs = net.forward()['prob'].flatten()
    feats = net.blobs[blob_name].data

    return list(probs), ([elem[0][0] for elem in feats[0].tolist()])
```

It was interesting for me to play with features values. I decided to plot covariance matrix for test set, that was sored by class. I applied the ***probs_feat*** function to all images from test set and using R ploted the matrixes. As it was expected features from almost last layes are pretty clustered!



```python
class0_6 = mpimg.imread('/home/veronika/materials/cv/cv_organizer/presentation/0_6.png')
plt.title("Covariance matrix for first 6 classes")
plt.imshow(class0_6)
```




    <matplotlib.image.AxesImage at 0x7fc1155c3290>




![png]((/img/post4/output_7_1.png)



```python
class7_11 = mpimg.imread('/home/veronika/materials/cv/cv_organizer/presentation/7_11.png')
plt.title("Covariance matrix for 7-11 classes")
plt.imshow(class7_11)
```




    <matplotlib.image.AxesImage at 0x7fc115bb6610>




![png]((/img/post4/output_8_1.png)



```python
class12_14 = mpimg.imread('/home/veronika/materials/cv/cv_organizer/presentation/12_14.png')
plt.title("Covariance matrix for 12-14 classes")
plt.imshow(class12_14)
```




    <matplotlib.image.AxesImage at 0x7fc115435990>




![png]((/img/post4/output_9_1.png)


Using R I plotted confusion matrix of predicted and target values of test set.


```python
confusion_m = mpimg.imread('/home/veronika/materials/cv/cv_organizer/presentation/confmatrix.png')
plt.imshow(confusion_m)
```




    <matplotlib.image.AxesImage at 0x7fc115341290>




![png]((/img/post4/output_11_1.png)


The last but not least fact is that in order to prevent overfitting all classes were splited into stritified folders (to save the distribution between classes in train and test sets) before training the model.


```python
train_dist = mpimg.imread('/home/veronika/materials/cv/cv_organizer/presentation/train.png')
plt.imshow(train_dist)

```




    <matplotlib.image.AxesImage at 0x7fc11578ccd0>




![png]((/img/post4/output_13_1.png)



```python
test_dist = mpimg.imread('/home/veronika/materials/cv/cv_organizer/presentation/test.png')
plt.imshow(test_dist)
```




    <matplotlib.image.AxesImage at 0x7fc1154dc190>




![png]((/img/post4/output_14_1.png)



