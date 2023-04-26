# CNN CNN what do you see ?

Neural networks are always refered to black-box because it is hard to understand individually each neuron's work and how they interact together.<br/>
Here we gonna open this black-box and give some visual explanations to understand how a convolutional neural network is able to "see" things.<br/>
To do so, we will work on maybe the most popular convolutional neural network (CNN), the VGG16. Despite its gae, it was first introduced in 2014[^1], it is still used in many cases and applications and keep producing mazing results compared to newer architectures.

#### Summary<a class="anchor" id=0></a>
[Method](#1) <br/>
[Resuts](#2)
  * [Image 1](#21)
    * [VGG16 pretrained](#211) | [VGG16 not trained](#212)
  * [Image 2](#22)
    * [VGG16 pretrained](#221) | [VGG16 not trained](#222)
    
[Observations](#3)<br/>
    


# Method<a class="anchor" id=1></a>

The VGG16 (Visual Geometry Group) is composed of 16 layers, 13 convolutional and 3 dense.[^2] 

<p align=center>
  <img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/VGG16.png?raw=true" width="60%" height="60%">
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

###### [summary](#0)
## Image 2<a class="anchor" id=22></a>

### Original image[^5] 
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/boat.jpg?raw=true" width="50%" height="50%">
</p>

### VGG16 pretrained<a class="anchor" id=221></a>

Output from the 1st convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/boat_model1_img_dim_224_224_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 2nd convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/boat_model1_img_dim_112_112_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 3rd convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/boat_model1_img_dim_56_56_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 4th convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/boat_model1_img_dim_28_28_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 5th convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/boat_model1_img_dim_14_14_12.png?raw=true" width="90%" height="90%">
</p>

### VGG16 not trained<a class="anchor" id=222></a>

Output from the 1st convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/boat_model2_img_dim_224_224_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 2nd convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/boat_model2_img_dim_112_112_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 3rd convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/boat_model2_img_dim_56_56_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 4th convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/boat_model2_img_dim_28_28_12.png?raw=true" width="90%" height="90%">
</p>

Output from the 5th convolutional block.
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/boat_model2_img_dim_14_14_12.png?raw=true" width="90%" height="90%">
</p>

# Observations<a class="anchor" id=3></a>
The images below are the aggregation of all the images displayed by each convolutional block.<br/>
For exemple, for the block 5, the 512 images have been aggregated together then normalized [0, 255] to be displayed.

<div>
<p align=center>
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/image1_comparison.png?raw=true" width="45%" height="45%">
<img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/image2_comparison.png?raw=true" width="45%" height="45%">
</p>
</div>


Note the difference between the pretrained and not trained VGG16: <br/>

On the 3 first blocks, the untrained model displays images that were less modified, closer from the original one.<br/>

It is because the kernels have issue to sort the information and prioritize the parts of the image that will allow the model to recognize identify it.<br/>
On the last 2 blocks, this lake of prioritization leads to blurry results.<br/><br/>

For the pretrained model on the opposite, the 2 first blocks quickly identify the sharps and edges of the image which helps the 3 last blocks to focus on the most pertinent areas.<br/><br/>

This allows us to conclude that a CNN trained with "imagenet" learns to identify patterns more than specific images.<br/>
This is why we can use these pretrained models (transfer-learning) and still have great results with images they were not trained on.

The model learns to see things in a way !

###### [summary](#0)

[^1]: [Very Deep Convolutional Networks for Large-Scale Image Recognition, by Karen Simonyan and Andrew Zisserman](https://arxiv.org/abs/1409.1556)
[^2]: [Photo credit: VGG16 by Gorlapraveen, Wikimedia commons](https://commons.wikimedia.org/wiki/File:VGG16.png)
[^3]: [Imagenet](https://www.image-net.org/)
[^4]: [Photo credit: Golden Retriever Dog by Karen Arnold, Public domain pictures](https://www.publicdomainpictures.net/en/view-image.php?image=437858&picture=golden-retriever-dog)
[^5]: [Photo credit: Fishing boat in the Canary Islands by Ian Sherlock, Wikimedia commons](https://commons.wikimedia.org/wiki/File:Fishing_boat_in_the_Canary_Islands.jpg)
