#!/usr/bin/env python
# coding: utf-8
##This file is specifically used for feeding the cropped 512x512 images to the model
##Will produce a file with the crater arrrays
## and the corresponding images 

# In[1]:

from src.acid import *    # The actual code is in src folder
from src.utils import *
import matplotlib
import matplotlib.pyplot as plt


# In[3]:


image_directory='/mnt/c/Users/Eulri/codings/europa/ACID/pluto_MPAN_cropped.jpg'  # directory of the image 
models_directory = '/mnt/c/Users/Eulri/codings/europa/acid2/ACID/models/*.h5' # directory of the models. Always ends with "*" .


image = Read_Preprocess_Image(image_directory,NORMALIZE=1, CONV_GS=1,INVERSE=0,EQUALIZE=0, CLAHE=1, 
                               RESIZE=0)


'''
model_inference is the main inference (predictions) function. It takes an image and the models directory as input.
Moreover, as ACID is an ensemble of 16 models, model_inference gives you the option of choosing only a subset of those.
Example: which_models = [1,2,3]. default is 'all'.
This function returns an objects master list with size [number_of_detections,11].
The 11 columns are respectively:['X_center_pix', 'Y_center_pix', 'R_pix', 'box_y1', 'box_x1', 'box_y2', 'box_x2', \
    'detection_thres','ellipticity','mask', 'object_size_pixels']
    
X_center_pix: The x coordinate of the center of the object, in pixel.
Y_center_pix: The y coordinate of the center of the object, in pixel.
R_pix: The radius of the object, in pixel.
box_** : The coordinates of the 2 diagonal points defining the bounding box.
detection_thres: The probability assigned to the detection by the model, from ~0 (not certain at all), to 1.
                  Exclusively choosing objects with a high detection_thres will lead to a conservative catalogue.

ellipticity: The ellipticity of the detected object. 1 is a perfect circle.
mask: a 512x512 binary mask. This is the shape of the detected object.
object_size_pixels: Total number of pixels of the mask.
'''
Craters_Master_list = model_inference(image,models_directory,which_models=[0,1,2,3,4])

#print(str("the list is ") + str(Craters_Master_list))
# In[ ]:





# In[4]:


"""
As ACID is an ensemble of 16 models, a lot of the detections will be duplicates. 
ACID hence contains 2 methods of filtering duplicates and generate 1 unique catalogue. 
get_unique_iou filter duplicates by calculating the intersection_over_union score.
get_unique_Longlat filter duplicates by looking for detections with similar location and size. 
    Please see Silburt, Ali-Dib et al. (2019) for more information, and for description of the parameters. 

"""
objects_unique = get_unique_iou(Craters_Master_list,iou_thres=0.5,detection_thres=0.75)
##CHANGED from detection_thres = 0.2
#objects_unique = get_unique_Longlat(Craters_Master_list,thresh_rad = 1.0,thresh_longlat2 = 1.8,detection_thres=0. )


# readable_output transforms the objects_unique (or Craters_Master_list) arrays into easy to read pandas dataframes
objects_unique_readable = readable_output(objects_unique)

# Additional filtering can be imposed on the dataframe.
objects_unique_readable__ = objects_unique_readable[(objects_unique_readable['detection_thres'] > 0.65) & \
                                                    (objects_unique_readable['ellipticity'] < 1.5) &\
                                                      (objects_unique_readable['object_size_pixels'] < 0.08*(512**2))]
##CHANGED from detection_thres = 0.2, size < 0.01*


# Calculate the sum of all masks as 1 512x512 image. This is a binary image containing all of the detections. 
totalmask = np.sum(objects_unique_readable__['mask'])
totalmask[totalmask>0] = 1

#%%

############ Plotting ##################
craters_image_read = plt.imread(image_directory)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(15,30)
plt.style.use('classic')
matplotlib.style='classic'

plt.subplot(1,2,1)
plt.imshow(cv2.resize(np.array(craters_image_read), (512, 512), interpolation=cv2.INTER_LINEAR) ,cmap='gray', origin = "lower")
plt.subplot(1,2,2)
plt.imshow(cv2.resize(np.array(craters_image_read), (512, 512), interpolation=cv2.INTER_LINEAR) ,cmap='gray', origin = 'lower')
totalmaskMasked = np.ma.masked_where(totalmask == 0, totalmask)
plt.imshow(totalmaskMasked,alpha=0.5,cmap='cool',origin = 'lower')

plt.savefig("")


# %%
