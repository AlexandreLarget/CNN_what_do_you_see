# CNN CNN what do you see ?

Neural networks are always refered to black-box because it is hard to understand individually each neuron's work and how they interact together.<br/>
Here we gonna open this black-box and give some visual explanations to understand how a convolutional neural network is able to "see" things.<br/>
To do so, we will work on maybe the most popular convolutional neural network (CNN), the VGG16. Despite its gae, it was first introduced in 2014[^1], it is still used in many cases and applications and keep producing mazing results compared to newer architectures.


## Quick sum up on the VGG16 and the CNN operating mode

The VGG16 (Visual Geometry Group) is composed of 16 layers, 13 convolutional and 3 dense.[^2] 

<p align=center>
  <img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/VGG16.png?raw=true" width="50%" height="50%">
</p>
  
We modify the model to produce 5 outputs (1 per convolutional block).

<p align=center>
  <img src="https://github.com/AlexandreLarget/CNN_what_do_you_see/blob/main/image/layer_ouputs.png?raw=true" width="50%" height="50%">
</p>

Since each neuron performs transformation by slidding it kernel through the image, the new outputs will allow us to see our image after those transformations and at several steps in the network.
  


[^1]: [Very Deep Convolutional Networks for Large-Scale Image Recognition, by Karen Simonyan and Andrew Zisserman](https://arxiv.org/abs/1409.1556)
[^2]: [Wikimedia commons](https://commons.wikimedia.org/wiki/File:VGG16.png)
