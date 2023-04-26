# CNN CNN what do you see ?

Neural networks are always refered to black-box because it is hard to understand individually each neuron's work and how they interact together.<br/>
Here we gonna open this black-box and give some visual explanations to understand how a convolutional neural network is able to "see" things.<br/>
To do so, we will work on maybe the most popular convolutional neural network (CNN), the VGG16. Despite its gae, it was first introduced in 2014[^1], it is still used in many cases and applications and keep producing mazing results compared to newer architectures.

#### Summary
[Method](#1) <br/>
[Resuts](#2) <br/>
  * [Image 1](#21)
    * [VGG16 pretrained](#211)
    * [VGG16 not trained](#212)
    * [comparison](#213)
   * [Image 2](#22)
    * [VGG16 pretrained](#221)
    * [VGG16 not trained](#222)
    * [comparison](#223)
   * [Image 3](#23)
    * [VGG16 pretrained](#231)
    * [VGG16 not trained](#232)
    * [comparison](#233)
    


# Method<a class="anchor" id=1></a>

The VGG16 (Visual Geometry Group) is composed of 16 layers, 13 convolutional and 3 dense.[^2] 

<p align=center>
  <img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/VGG16.png?raw=true" width="50%" height="50%">
</p>
  
We modify the model to produce 5 outputs (1 per convolutional block).

<p align=center>
  <img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/layer_ouputs.png?raw=true" width="50%" height="50%">
</p>

Since each neuron performs transformation by slidding it kernel through the image, the new outputs will allow us to see our image after those transformations and at several steps in the network.<br/>
We will use **2 VGG16, one pretrained on "imagenet"[^3], the second not trained**, so with random normalized weights.<br/>
We will compare visually the transformations performed by the 2 models and see how the training affects these transformations.
  
  
# Results<a class="anchor" id=2></a>

For each image, we will display 12 output images per convolutional block (images are selected by sorting the sum of their matrix).

## Image 1<a class="anchor" id=21></a>

### Original image[^4] 
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden.jpg?raw=true" width="25%" height="25%">
</p>

### VGG16 pretrained<a class="anchor" id=211></a>

Output from the 1st convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model1_img_dim_224_224_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 2nd convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model1_img_dim_112_112_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 3rd convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model1_img_dim_56_56_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 4th convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model1_img_dim_28_28_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 5th convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model1_img_dim_14_14_12.png?raw=true" width="90%" height="90%">
</p>

### VGG16 not trained<a class="anchor" id=212></a>

Output from the 1st convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model2_img_dim_224_224_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 2nd convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model2_img_dim_112_112_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 3rd convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model2_img_dim_56_56_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 4th convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model2_img_dim_28_28_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 5th convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model2_img_dim_14_14_12.png?raw=true" width="90%" height="90%">
</p>

### Comparison<a class="anchor" id=213></a>

<div>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model1_aggregated_img_dim_224_224.png?raw=true" width="50%" height="50%">
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/golden_model2_aggregated_img_dim_224_224.png?raw=true" width="50%" height="50%">
</div>


[^1]: [Very Deep Convolutional Networks for Large-Scale Image Recognition, by Karen Simonyan and Andrew Zisserman](https://arxiv.org/abs/1409.1556)
[^2]: [Photo credit: Gorlapraveen, Wikimedia commons](https://commons.wikimedia.org/wiki/File:VGG16.png)
[^3]: Imagenet
[^4]: [Photo credit: Karen Arnold, Public domain pictures](https://www.publicdomainpictures.net/en/view-image.php?image=437858&picture=golden-retriever-dog)
