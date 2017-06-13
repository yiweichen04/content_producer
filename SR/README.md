The demo file for super resolution is demo.ipynb.

In demo.ipynb, there are two important funcitons. The first one is for model 
intialization. It would load model from './checkpoint'.

The second function is the super resolution function. You can set the input 
image path by yourself. This function would save LR(Low resolution) images 
and SR(Super resolution) images to './output/'. On the other hand, it would also
 return the SR image as numpy array for following applications.   

