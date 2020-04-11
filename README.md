# DeepAPS 

### Estimating conformational traits in dairy cattle: a deep learning approach

Code by Laura Zingaretti with contributions from Jessica Nye. 

### Citation:

Nye J, Zingaretti LM, Pérez-Enciso, M Estimating conformational traits in daity cattle with DeepAPS: a two-step Deep Learning Automated Phenotyping and Segmentation approach. Submitted to Frontiers in Genetics

Assessing conformation features in an accurate and rapid manner remains a challenge in the dairy industry. While recent
developments in computer vision has greatly improved automated background removal, these methods have not been fully
translated to biological studies. Here, we present a composite method (DeepAPS) that combines two readily available algorithms, Mask R-RNN (https://github.com/matterport/Mask_RCNN) and Kanezaki's (https://github.com/kanezaki/pytorch-unsupervised-segmentation) in
order to create a precise mask for an animal image. This method performs accurately when compared with manual classification
of proportion of coat color with an adjusted R2 = 0.926. Using the output mask, we are able to automatically extract useful
phenotypic information for fourteen additional morphological features. Using pedigree and image information from a web catalog
(www.semex.com), we estimated high heritabilities (ranging from h2 = 0.18 – 0.82), indicating that meaningful biological
information has been extracted automatically from imaging data. This method can be applied to other datasets and requires only
a minimal number of image annotations (~50) to train this partially supervised machine-learning approach. DeepAPS allows for the
rapid and accurate quantification of multiple phenotypic measurements while minimizing study cost.

### Contact: 

m.lau.zingaretti@gmail.com



* We strongly recommend to use a conda environment to run this code.

How you should proceed? 

Open bash and type: 


```console
laura@localhost:~$ conda create --name cows python=3.6 -y
laura@localhost:~$ conda activate cows 
laura@localhost:~$ conda install -c anaconda ipykernel
laura@localhost:~$ python -m ipykernel install --user --name=cows
```

Just check your Jupyter Notebook, to see the shining cows and chose this as notebook.

### 1. Installing dependences 

Note that you can do this step from bash using conda utilities. 

To install dependences from bash, just type: 

```console
laura@localhost:~$ conda activate cows
laura@localhost:~$ conda install -c anaconda pytorch-gpu tensorflow-gpu=1.13.1
laura@localhost:~$ conda install -c conda-forge keras=2.1 matplotlib python-utils
laura@localhost:~$ conda install -c anaconda pandas
laura@localhost:~$ conda install pillow
laura@localhost:~$ conda install -c anaconda scikit-image
laura@localhost:~$ conda install -c intel scikit-learn
laura@localhost:~$ conda install -c auto python-utils
laura@localhost:~$ conda install -c conda-forge imgaug
laura@localhost:~$ pip install cython
laura@localhost:~$ pip install pycocotools
laura@localhost:~$ pip install utils
```

*Warning*: Note that these instructions run the code using CUDA-GPU toolkit. Please install tensorflow-cpu and torch-cpu if you don't have a nvidia graphic card. 


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import sys
import os
import numpy as np
import torch.nn.init
import utils
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import skimage.io
from skimage import morphology
from scipy import ndimage
from skimage import morphology
from skimage.feature import peak_local_max
```


```python
# Load image
Orig=cv2.imread("/home/laura/images/FR2238143895.jpg")
```


```python
# Show image
plt.imshow(Orig)
```




    <matplotlib.image.AxesImage at 0x7f2c0e848eb8>




![png](output_5_1.png)


### 2. Download Mask_RCNN 
go to https://github.com/matterport/Mask_RCNN and download repo 



```python
!git clone https://github.com/matterport/Mask_RCNN.git
```

    Cloning into 'Mask_RCNN'...
    remote: Enumerating objects: 956, done.[K
    remote: Total 956 (delta 0), reused 0 (delta 0), pack-reused 956[K
    Receiving objects: 100% (956/956), 111.82 MiB | 18.65 MiB/s, done.
    Resolving deltas: 100% (568/568), done.



```python
#check the path 
#print('\n'.join(sys.path))
```

### 3. Load pre-trained weights from coco database (containing 80 categories)
- Note that the following instruction works in linux/mac to download the .h5 file automatically.
- If you are in a windows machine, please go to the link and downoad the dataset by hand. 
- We save mask_rcnn_coco.h5 in /home/laura/Mask_RCNN/samples/coco/mask_rcnn_coco.h5 folder, which should be the COCO_MODEL_PATH


```python
!wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
```

    --2020-04-11 17:51:09--  https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
    Resolving github.com (github.com)... 140.82.118.4
    Connecting to github.com (github.com)|140.82.118.4|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://github-production-release-asset-2e65be.s3.amazonaws.com/107595270/872d3234-d21f-11e7-9a51-7b4bc8075835?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20200411%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200411T155110Z&X-Amz-Expires=300&X-Amz-Signature=1417db0979de70322e32361342271b0ea9f1e3ad819a62a113ef0391474b7a38&X-Amz-SignedHeaders=host&actor_id=0&repo_id=107595270&response-content-disposition=attachment%3B%20filename%3Dmask_rcnn_coco.h5&response-content-type=application%2Foctet-stream [following]
    --2020-04-11 17:51:10--  https://github-production-release-asset-2e65be.s3.amazonaws.com/107595270/872d3234-d21f-11e7-9a51-7b4bc8075835?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20200411%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200411T155110Z&X-Amz-Expires=300&X-Amz-Signature=1417db0979de70322e32361342271b0ea9f1e3ad819a62a113ef0391474b7a38&X-Amz-SignedHeaders=host&actor_id=0&repo_id=107595270&response-content-disposition=attachment%3B%20filename%3Dmask_rcnn_coco.h5&response-content-type=application%2Foctet-stream
    Resolving github-production-release-asset-2e65be.s3.amazonaws.com (github-production-release-asset-2e65be.s3.amazonaws.com)... 52.217.42.44
    Connecting to github-production-release-asset-2e65be.s3.amazonaws.com (github-production-release-asset-2e65be.s3.amazonaws.com)|52.217.42.44|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 257557808 (246M) [application/octet-stream]
    Saving to: 'mask_rcnn_coco.h5'
    
    100%[======================================>] 257,557,808 1.17MB/s   in 64s    
    
    2020-04-11 17:52:15 (3.81 MB/s) - 'mask_rcnn_coco.h5' saved [257557808/257557808]
    



```python
## Add repo to sys path 
sys.path.append(os.path.join("/home/laura/Mask_RCNN/"))  # To find local version of the library
import mrcnn.model as modellib

## Import MS-COCO from Lin et al., 2014
## Change for custom training set
sys.path.append(os.path.join("/home/laura/Mask_RCNN/samples/coco/"))  # To find local version
#this is to find coco.py

## Directory to save logs and trained model
MODEL_DIR = os.path.join("/home/laura/Mask_RCNN/logs")

## Local path to trained weights file
COCO_MODEL_PATH = os.path.join("/home/laura/Mask_RCNN/samples/coco/mask_rcnn_coco.h5")



```

    /usr/anaconda3/envs/DeepAPS/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /usr/anaconda3/envs/DeepAPS/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /usr/anaconda3/envs/DeepAPS/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /usr/anaconda3/envs/DeepAPS/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /usr/anaconda3/envs/DeepAPS/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /usr/anaconda3/envs/DeepAPS/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])
    Using TensorFlow backend.



```python
import coco
## Adapted from Mask_RCNN He et al., 2017
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

## MS-COCO class names from Lin et al., 2014
# coco has 80 objects
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

## Create model object in inference mode.
model_Coco = modellib.MaskRCNN(mode = "inference", model_dir = MODEL_DIR, config = config)

## Load weights trained on MS-COCO from Lin et al., 2014
model_Coco.load_weights(COCO_MODEL_PATH, by_name = True)

```

    
    Configurations:
    BACKBONE                       resnet101
    BACKBONE_STRIDES               [4, 8, 16, 32, 64]
    BATCH_SIZE                     1
    BBOX_STD_DEV                   [0.1 0.1 0.2 0.2]
    COMPUTE_BACKBONE_SHAPE         None
    DETECTION_MAX_INSTANCES        100
    DETECTION_MIN_CONFIDENCE       0.7
    DETECTION_NMS_THRESHOLD        0.3
    FPN_CLASSIF_FC_LAYERS_SIZE     1024
    GPU_COUNT                      1
    GRADIENT_CLIP_NORM             5.0
    IMAGES_PER_GPU                 1
    IMAGE_CHANNEL_COUNT            3
    IMAGE_MAX_DIM                  1024
    IMAGE_META_SIZE                93
    IMAGE_MIN_DIM                  800
    IMAGE_MIN_SCALE                0
    IMAGE_RESIZE_MODE              square
    IMAGE_SHAPE                    [1024 1024    3]
    LEARNING_MOMENTUM              0.9
    LEARNING_RATE                  0.001
    LOSS_WEIGHTS                   {'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 1.0, 'mrcnn_mask_loss': 1.0}
    MASK_POOL_SIZE                 14
    MASK_SHAPE                     [28, 28]
    MAX_GT_INSTANCES               100
    MEAN_PIXEL                     [123.7 116.8 103.9]
    MINI_MASK_SHAPE                (56, 56)
    NAME                           coco
    NUM_CLASSES                    81
    POOL_SIZE                      7
    POST_NMS_ROIS_INFERENCE        1000
    POST_NMS_ROIS_TRAINING         2000
    PRE_NMS_LIMIT                  6000
    ROI_POSITIVE_RATIO             0.33
    RPN_ANCHOR_RATIOS              [0.5, 1, 2]
    RPN_ANCHOR_SCALES              (32, 64, 128, 256, 512)
    RPN_ANCHOR_STRIDE              1
    RPN_BBOX_STD_DEV               [0.1 0.1 0.2 0.2]
    RPN_NMS_THRESHOLD              0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE    256
    STEPS_PER_EPOCH                1000
    TOP_DOWN_PYRAMID_SIZE          256
    TRAIN_BN                       False
    TRAIN_ROIS_PER_IMAGE           200
    USE_MINI_MASK                  True
    USE_RPN_ROIS                   True
    VALIDATION_STEPS               50
    WEIGHT_DECAY                   0.0001
    
    
    WARNING:tensorflow:From /usr/anaconda3/envs/DeepAPS/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From /home/laura/Mask_RCNN/mrcnn/model.py:772: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.


### 3. Call functions.py from code folder 



```python
sys.path.append(os.path.join("/home/laura/code/"))  # To find local version of the library
from functions import *
```

### 4. Run Mask-RCNN to object detection 


```python
image = skimage.io.imread('/home/laura/images/FR2238143895.jpg')
```


```python
## Run Object detection using coco
results = model_Coco.detect([image], verbose = 1)

## Visualize results
r = results[0]

## Select only class id you are interested in (we have chosen cow n=20, to change object the 20 in the next two lines should be changed)
if (20 in r['class_ids']):
    w = r['class_ids'].tolist().index(20)
    im = image[r['rois'][w].item(0):r['rois'][w].item(2), r['rois'][w].item(1):r['rois'][w].item(3),]

    ## Select only region of interest for desired object
    A = np.transpose(r['masks'], [0,1,2])
    A = A.astype(int)
    A = A[r['rois'][w].item(0):r['rois'][w].item(2), r['rois'][w].item(1):r['rois'][w].item(3),0]

```

    Processing 1 images
    image                    shape: (749, 1000, 3)        min:    0.00000  max:  255.00000  uint8
    molded_images            shape: (1, 1024, 1024, 3)    min: -123.70000  max:  151.10000  float64
    image_metas              shape: (1, 93)               min:    0.00000  max: 1024.00000  float64
    anchors                  shape: (1, 261888, 4)        min:   -0.35390  max:    1.29134  float32



```python
# im contains the box (RoI) with cow
# note most of the background is out
plt.imshow(im)
```




    <matplotlib.image.AxesImage at 0x7f2bb42f7ba8>




![png](output_18_1.png)



```python
# CUDA is STRONGLY recommended
use_cuda = False #when cuda is not available
#use_cuda = torch.cuda.is_available()
```


```python
data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]))
if use_cuda:
    data=data.cuda()
```


```python
## Unsupervised refinement adapted from Kanezaki, 2018 https://github.com/kanezaki/pytorch-unsupervised-segmentation
## SLIC from Achanta et al., 2012
nConv=3 #you can choose more or less conv
lr= 0.2 # lr you can change this parameter 
maxIter=200 #maximum number of iteractions 
minLabels=3 #minimun number of cluster
visualize=True
labels = A
labels = labels.reshape(im.shape[0]*im.shape[1])
u_labels = np.unique(A)
l_inds = []

for i in range(len(u_labels)):
    l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )
    
    
    torch.cuda.empty_cache()
    model = MyNet( data.size(1) )
    if use_cuda:
        model.cuda()
        for i in range(nConv-1):
            model.conv2[i].cuda()
            model.bn2[i].cuda()
    model.train()
       
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
    label_colours = np.random.randint(255,size = (100,3))
        
    for batch_idx in range(maxIter):
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, 100 )
        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        if visualize:
            im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[l_inds[ i ]] = u_labels_per_sp[ np.argmax( hist ) ]
        target = torch.from_numpy( im_target )
        if use_cuda:
            target = target.cuda()
        target = Variable( target ,volatile=True)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if nLabels <= minLabels:
            print ("nLabels", nLabels, "reached minLabels", minLabels, ".")
            break
            
        ## Save output image
        if not visualize:
            output = model( data )[ 0 ]
            output = output.permute( 1, 2, 0 ).contiguous().view( -1, 100 )
            ignore, target = torch.max( output, 1 )
            im_target = target.data.cpu().numpy()
            im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
        #cv2.imwrite(Outdir + "raw_Mask_" + file_name, im_target_rgb)

```

    /usr/anaconda3/envs/DeepAPS/lib/python3.6/site-packages/ipykernel_launcher.py:50: UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.


    nLabels 2 reached minLabels 3 .



```python
# show output
plt.imshow(im_target_rgb)
```




    <matplotlib.image.AxesImage at 0x7f2bb425a3c8>




![png](output_22_1.png)


### 5. Write image to file  


```python
#cv2.imwrite(Outdir + "raw_Mask_" + file_name, im_target_rgb)

```

### 6. Cleaning image 


```python
#Applying mediaBlur filtering to clean the output 
# median smoothing 
im_target_rgb = cv2.medianBlur(im_target_rgb, 7)
plt.imshow(im_target_rgb)
#write final mask to file 
#cv2.imwrite(Outdir + "/Final_Mask_" + file_name, cv2.medianBlur(im_target_rgb, 5))

```




    <matplotlib.image.AxesImage at 0x7f2badf7b2e8>




![png](output_26_1.png)


### 7. Image segmentation 

After the smoothing, we are ready to see how many pixels correspond to background and foreground implementing a simple kMeans algorithm.  


```python
## Segment image
Seg = cv2.medianBlur(im_target_rgb, 5)
image = Seg.reshape((Seg.shape[0] * Seg.shape[1], 3))
clt = KMeans(n_clusters = nLabels)
clt.fit(image)

hist = centroid_histogram(clt)
bar,percent = plot_colors(hist, clt.cluster_centers_)
(cenx, ceny) = (im.shape[1]/2, im.shape[0]/2)
# obtain the central pixel 
central=im_target_rgb[int(round(ceny)),int(round(cenx)),:]
```


```python
## Apply mask
imga = np.zeros([Seg.shape[0],Seg.shape[1],3], np.uint8)
## Color applied can be changed to be any color Grey = (200, 200, 200), Black = (0, 0, 0), White = (255, 255, 255)
imga[:] = (255, 0, 0)
m1 = np.all(im_target_rgb ==  central, axis = -1)
imga[m1] = im[m1]
dif = imga-im

image = imga.reshape((imga.shape[0] * imga.shape[1], 3))
clt = KMeans(n_clusters = 3)
clt.fit(image)
hist1 = centroid_histogram(clt)
bar1,percent1 = plot_colors(hist1, clt.cluster_centers_)

```


```python
clt.n_clusters
```




    3




```python
# three color clusters
plt.imshow(bar1) 
```




    <matplotlib.image.AxesImage at 0x7f2ae43654e0>




![png](output_31_1.png)



```python
# mask in black, object minus background
plt.imshow(dif) 
```




    <matplotlib.image.AxesImage at 0x7f2badf1a278>




![png](output_32_1.png)



```python
# the original image into a homogeneous background 
plt.imshow(imga[:,:,::-1]) 
#cv2.imwrite( Outdir + "/applied_Mask_" + file_name, imga[:,:,::-1] )

```




    <matplotlib.image.AxesImage at 0x7f2c3ba2e908>




![png](output_33_1.png)



```python
# a regresentation of the percentage of background/ foreground 
plt.imshow(bar) 
```




    <matplotlib.image.AxesImage at 0x7f2c3ba13470>




![png](output_34_1.png)


### Calculating black color percentage


```python
A_colo = np.round(np.sum(clt.cluster_centers_, axis = 1),decimals = 0)
index = int(np.argwhere(A_colo == 255))
percent1 = list(percent1)
## Remove masked portion of image
percent1.pop(index)
suma = np.sum(percent1)
porcent = []

## Calculate proportion of color clusters
for i in  percent1:
    porcent.append(100*i/suma)

print(porcent) # 60 black ~40 white
```

    [60.28039394449249, 39.71960605550752]


### 8.Transform final mask into 2-D black and white image


```python
gray = cv2.cvtColor(im_target_rgb, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

## Extract the outline of the object
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations = 1)
edged = cv2.erode(edged, None, iterations = 1)

## Clean image - values should be optimized
processed = morphology.remove_small_objects(edged.astype(bool), min_size = 300, connectivity = 1000).astype(int)

## transform the image from a T/F matrix to a B/W image
out = processed*255

## calcualte size of image
height, width = out.shape
total_pixels = out.size

## calculate number of pixels in the perimeter
outline = cv2.countNonZero(out)
total_perimeter = outline / total_pixels
```

### 9. Example of feature extraction
The following codes contain some examples about morphological features extraction from images 


```python
height #see height
```




    397




```python
width #see body width
```




    527




```python
#total number of pixels 
total_pixels
```




    209219




```python
total_perimeter  ### proportion of pixels in the perimeter 
```




    0.011714997203886836




```python
## Export the outline of the object
## cv2.imwrite(Outdir + "object_outline_" + file_name, out)
y, x = np.nonzero(out)     
```


```python
## Example of feature extraction - bottom of back leg
backleg_x = []
backleg_y = []
for i in range(len(y)):
    if y[i] > max(y)- height/5:
        if x[i] < min(x) + width/5:
            backleg_x.append(x[i])
            backleg_y.append(y[i])
if backleg_x:
    bl_index = np.argmax(backleg_y)

```


```python
## Example of feature extraction - bottom of front leg
frontleg_x = []
frontleg_y = []
for i in range(len(y)):
    if y[i] > max(y) - height/5:
        if x[i] > max(x) - width/2:
            frontleg_x.append(x[i])
            frontleg_y.append(y[i])
if frontleg_x:
    fl_index = np.argmax(frontleg_y)
           
```


```python
## Example of distance calculation
if frontleg_x and backleg_x:
    len_gait = np.sqrt(((frontleg_x[fl_index] - backleg_x[bl_index])**2)+((frontleg_y[fl_index] - backleg_y[bl_index])**2))

```


```python
len_gait ## distance 
```




    318.8306760648981




```python
(cenx, ceny) = (im.shape[1]/2, im.shape[0]/2) 

central=im_target_rgb[int(round(ceny)),int(round(cenx)),:]


# show our color bar
imga = np.zeros([Seg.shape[0],Seg.shape[1],3], np.uint8)
imga[:] = (200, 200, 200)

m1=np.all(im_target_rgb ==  central, axis=-1)

imga[m1]=im[m1]
dif= imga-im

```


```python
### find local maxima
coordinates = peak_local_max(imga, min_distance=20)

### transform image into 2-D black and white image
gray = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

### only keep the outline of the cow
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

### clean the islands from the image
processed = morphology.remove_small_objects(edged.astype(bool), min_size=300, connectivity=1000).astype(int)

### transform the image from a T/F matrix to a B/W image
out = processed*255

### calcualte size of image
height, width = out.shape
total_pixels = out.size

### calculate number of pixels in the perimeter	
outline = cv2.countNonZero(out)
total_perimeter = outline / total_pixels
```


```python
y, x = np.nonzero(out)
### find bottom of back leg
backleg_x=[]
backleg_y=[]
for i in range(len(y)):
    if y[i] > max(y)-height/5:
        if x[i] < min(x)+width/5:
            backleg_x.append(x[i])
            backleg_y.append(y[i])
    if backleg_x:
        bl_index=np.argmax(backleg_y)

### find bottom of front leg
frontleg_x=[]
frontleg_y=[]
for i in range(len(y)):
    if y[i] > max(y)-height/5:
        if x[i] > max(x)-width/2:
            frontleg_x.append(x[i])
            frontleg_y.append(y[i])
    if frontleg_x:
        fl_index=np.argmax(frontleg_y)


```


```python
### find rear
rear_x=[]
rear_y=[]
for i in range(len(y)):
    if y[i] < height/4:
        if x[i] < min(x) + width/11:
            rear_x.append(x[i])
            rear_y.append(y[i])
    if rear_x:
        r_index=np.argmin(rear_x)
```


```python
### find middle of the stomach
belly_x=[]
belly_y=[]
for i in range(len(y)):
    if height-height/4 > y[i] > height/2:
        if width/4 < x[i] < width/2:
            belly_x.append(x[i])
            belly_y.append(y[i])
    if belly_x:
        tip_index=np.argmax(belly_y)

### find back
mback_y=[]
for i in range(len(y)):
    if x[i] == belly_x[tip_index]:
        mback_y.append(y[i])
    if mback_y:
        mb_index=np.argmin(mback_y)

### find top of back legs
bleg_x=[]
bleg_y=[]
for i in range(len(y)):
    if y[i] > height/2:
        if belly_x:
            if belly_x[tip_index] > x[i] > belly_x[tip_index]-width/5:
                bleg_x.append(x[i])
                bleg_y.append(y[i])
        if bleg_x:
            bl_top_index=np.argmin(bleg_y)

### find back
bback_y=[]
for i in range(len(y)):
    if x[i] == bleg_x[bl_top_index]:
        bback_y.append(y[i])
        if bback_y:
            bb_index=np.argmin(bback_y)

            ### find top of front legs
fleg_x=[]
fleg_y=[]
for i in range(len(y)):
    if y[i] > height/2:
        if belly_x and frontleg_x:
            if belly_x[tip_index] < x[i] < belly_x[tip_index]+width/3:
                if x[i] < frontleg_x[fl_index]:
                    fleg_x.append(x[i])
                    fleg_y.append(y[i])
    if fleg_x:
        fl_top_index=np.argmin(fleg_y)

### find back
fback_y=[]
for i in range(len(y)):
    if x[i] == fleg_x[fl_top_index]:
        fback_y.append(y[i])
        if fback_y:
            fb_index=np.argmin(fback_y)
head_x=[]
head_y=[]

for i in range(len(y)):
    if y[i] < min(y)+height/4:
        if x[i] > max(x)-width/3:
            head_x.append(x[i])
            head_y.append(y[i])
    if head_x:
        ear_index=np.argmin(head_y)
```


```python
### find neck from small head
Shead_x=[]
Shead_y=[]
for i in range(len(y)):
    if y[i] < min(y)+height/4:
        if x[i] > max(x)-width/3:
            Shead_x.append(x[i])
            Shead_y.append(y[i])
        if Shead_x:
            Sbneck_index=np.argmin(Shead_x)
            Sfneck_index=np.argmax(Shead_y)

```


```python
### distance between legs
if frontleg_x and backleg_x:
    len_gait = np.sqrt(((frontleg_x[fl_index] - backleg_x[bl_index])**2)+((frontleg_y[fl_index] - backleg_y[bl_index])**2))
```


```python
len_gait
```




    470.7706447942565




```python
### area of body triangle between rear, middle stomach, and back of neck
if Shead_x and belly_x and rear_x:
    tail_tip = np.sqrt(((belly_x[tip_index] - rear_x[r_index])**2)+((belly_y[tip_index] - rear_y[r_index])**2))
    neck_tip = np.sqrt(((belly_x[tip_index] - Shead_x[Sbneck_index])**2)+((belly_y[tip_index] - Shead_y[Sbneck_index])**2))
    neck_tail = np.sqrt(((rear_x[r_index] - Shead_x[Sbneck_index])**2)+((rear_y[r_index] - Shead_y[Sbneck_index])**2))
    perim = 0.5*(tail_tip + neck_tip + neck_tail)
    cow_triangle = np.sqrt(perim * (perim - tail_tip) * (perim - neck_tip) * (perim - neck_tail))

### area of body polygon between rear, middle stomach, front of neck, and back of neck
cow_poly = np.abs((((belly_x[tip_index]*Shead_y[Sfneck_index])-(belly_y[tip_index]*Shead_x[Sfneck_index]))+((Shead_x[Sfneck_index]*Shead_y[Sbneck_index])-(Shead_y[Sfneck_index]*Shead_x[Sbneck_index]))+((Shead_x[Sbneck_index]*rear_y[r_index])-(Shead_y[Sbneck_index]*rear_x[r_index]))+((rear_x[r_index]*belly_y[tip_index])-(rear_y[r_index]*belly_x[tip_index])))/2)
```


```python
# Perimeter
perim
```




    707.1557276217562



### Add point of morphological measures to image 


```python

Z=cv2.circle(im_target_rgb,(backleg_x[bl_index],backleg_y[bl_index]), 5, (255,255,255), -1)
Z=cv2.circle(Z,(frontleg_x[fl_index],frontleg_y[fl_index]), 5, (255,255,255), -1)
Z=cv2.circle(Z,(rear_x[r_index],rear_y[r_index]), 5, (255,255,255), -1)
Z=cv2.circle(Z,(head_x[ear_index],head_y[ear_index]), 5, (255,255,255), -1)
Z=cv2.circle(Z,(belly_x[tip_index],belly_y[tip_index]), 5, (255,255,255), -1)
Z=cv2.circle(Z,(bleg_x[bl_top_index],bleg_y[bl_top_index]), 5, (255,255,255), -1)
Z=cv2.circle(Z,(fleg_x[fl_top_index],fleg_y[fl_top_index]), 5, (255,255,255), -1)

plt.imshow(Z)
```




    <matplotlib.image.AxesImage at 0x7f2ab838d4a8>




![png](output_60_1.png)


# Good luck!


```python

```

