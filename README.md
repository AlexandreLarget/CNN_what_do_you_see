# CNN CNN what do you see ?

Neural networks are always refered to black-box because it is hard to understand individually each neuron's work and how they interact together.<br/>
Here we gonna open this black-box and give some visual explanations to understand how a convolutional neural network is able to "see" things.<br/>
To do so, we will work on maybe the most popular convolutional neural network (CNN), the VGG16. Despite its gae, it was first introduced in 2014[^1], it is still used in many cases and applications and keep producing mazing results compared to newer architectures.


## Method

The VGG16 (Visual Geometry Group) is composed of 16 layers, 13 convolutional and 3 dense.[^2] 

<p align=center>
  <img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/VGG16.png?raw=true" width="50%" height="50%">
</p>
  
We modify the model to produce 5 outputs (1 per convolutional block).

<p align=center>
  <img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/layer_ouputs.png?raw=true" width="50%" height="50%">
</p>

Since each neuron performs transformation by slidding it kernel through the image, the new outputs will allow us to see our image after those transformations and at several steps in the network.
  
  
# Results

For each image, we will display 12 output images per convolutional block (images are selected by sorting the sum of their matrix).

## Image 1

#### Original image[^3] 
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden.jpg?raw=true" width="25%" height="25%">
</p>

#### Output from the 1st convolutional block.
<p align=center>
Image dimensions: 224x224 px <br/>
Number of output images at this step: 64
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model1_img_dim_224_224_12.png?raw=true" width="100%" height="100%">
</p>

#### Output from the 2st convolutional block.
<p align=center>
Image dimensions: 112x112 px <br/>
Number of output images at this step: 128
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model1_img_dim_112_112_12.png?raw=true" width="100%" height="100%">
</p>

#### Output from the 1st convolutional block.
<p align=center>
Image dimensions: 224x224 px <br/>
Number of output images at this step: 64
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model1_img_dim_224_224_12.png?raw=true" width="100%" height="100%">
</p>

#### Output from the 1st convolutional block.
<p align=center>
Image dimensions: 224x224 px <br/>
Number of output images at this step: 64
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model1_img_dim_224_224_12.png?raw=true" width="100%" height="100%">
</p>

#### Output from the 1st convolutional block.
<p align=center>
Image dimensions: 224x224 px <br/>
Number of output images at this step: 64
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model1_img_dim_224_224_12.png?raw=true" width="100%" height="100%">
</p>


[^1]: [Very Deep Convolutional Networks for Large-Scale Image Recognition, by Karen Simonyan and Andrew Zisserman](https://arxiv.org/abs/1409.1556)
[^2]: [Photo credit: Gorlapraveen, Wikimedia commons](https://commons.wikimedia.org/wiki/File:VGG16.png)
[^3]: [Photo credit: Karen Arnold, Public domain pictures](https://www.publicdomainpictures.net/en/view-image.php?image=437858&picture=golden-retriever-dog)
