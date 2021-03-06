import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
import math
import os
import cv2

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    
    cell_num = 10*10+5*5+3*3+1*1
    box_num = 4*cell_num
    small_scale = np.array(small_scale)
    large_scale = np.array(large_scale)
    
    boxes = [] 
    #boxes = np.zeros((cell_num, 4, 8)) # [number of cells, default bounding boxes in each cell, attributes in each bounding box]
    # 4: [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)]
    # 8: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    #grid_size_group = [10,5,3,1]
    
    for grid_size in layers:
        
        k = layers.index(grid_size)
        
        box_width = np.array([small_scale[k], large_scale[k], large_scale[k]*np.sqrt(2), large_scale[k]/np.sqrt(2)]) # 4*1
        box_height = np.array([small_scale[k], large_scale[k], large_scale[k]/np.sqrt(2), large_scale[k]*np.sqrt(2)]) # 4*1
        
        box = np.zeros((grid_size*grid_size, 4, 8))
        ## generate bounding boxes in each cell 
        for i in range(grid_size): 
            for j in range(grid_size):
                
                #x_center = ((i+1)/2)/grid_size # 1/(grid_size*2) + i 
                #y_center = ((j+1)/2)/grid_size
                
                x_center = 1/(grid_size*2) + j/grid_size
                y_center = 1/(grid_size*2) + i/grid_size
                
                x_min = x_center - box_width/2 # 4*1
                x_max = x_center + box_width/2 # 4*1
                
                y_min = y_center - box_height/2
                y_max = y_center + box_height/2
                
                # clip boxes
                x_min[x_min < 0] = 0
                y_min[y_min < 0] = 0
                
                x_max[x_max > 1] = 1
                y_max[y_max > 1] = 1
                
                box_width[box_width > 1] = 1
                box_height[box_height > 1] = 1
                
                # create center matrix
                x_c = np.zeros((4))
                y_c = np.zeros((4))
                
                x_c[:] = x_center
                y_c[:] = y_center
                
                box[i*grid_size+j,:,0] = x_c
                box[i*grid_size+j,:,1] = y_c
                box[i*grid_size+j,:,2] = box_width
                box[i*grid_size+j,:,3] = box_height
                box[i*grid_size+j,:,4] = x_min
                box[i*grid_size+j,:,5] = y_min
                box[i*grid_size+j,:,6] = x_max
                box[i*grid_size+j,:,7] = y_max
                
        #boxes = np.concatenate((boxes, box, axis = 0))
        boxes.append(box)

    # reshape boxes to [box_num, 8]
    # todo
    boxes = np.concatenate((boxes[0], boxes[1], boxes[2], boxes[3]), axis = 0)
    boxes = boxes.reshape((box_num, 8))
    
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    ious_true = ious>threshold
    
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    
    # p - ann_box
    p = boxs_default[ious_true]
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    if len(p) == 0:
        ious_true = np.argmax(ious)
        p = boxs_default[ious_true]
        p = p.reshape((1,8))
        
    gx = x_min + (x_max - x_min)/2
    gy = y_min + (y_max - y_min)/2
    gw = x_max - x_min
    gh = y_max - y_min
    
    relative_center_x = (gx - p[:,0])/p[:,2]
    relative_center_y = (gy - p[:,1])/p[:,3]
    
    relative_center_x[relative_center_x < 0] = 0
    relative_center_x[relative_center_x > 1] = 1
    
    relative_center_y[relative_center_y < 0] = 0
    relative_center_y[relative_center_y > 1] = 1
    
    relative_width = np.log(gw/p[:,2])
    relative_height = np.log(gh/p[:,3])
    
    ann_box[ious_true,:] = np.array([relative_center_x, relative_center_y, 
                                     relative_width, relative_height]).T # [540,4]  
    
    ann_confidence[ious_true,cat_id] = 1 # [540,4] one hot vectors
    ann_confidence[ious_true,-1] = 0

    return ann_box, ann_confidence



class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        
        #notice:
        #you can split the dataset into 80% training and 20% testing here, by slicing self.img_names with respect to self.train    
        
        if self.train == True:
            self.img_names = self.img_names[slice(0, int(0.8*len(self.img_names)), 1)]
        
        elif self.train == False:
            self.img_names = self.img_names[slice(int(0.8*len(self.img_names)), -1, 1)]


    def __len__(self):
        if self.train == True:
            return len(self.img_names)*2
        elif self.train == False:
            return len(self.img_names)
        

    def __getitem__(self, index):
        img_length = len(self.img_names)
        if index < img_length:
            ann_box = np.zeros([self.box_num,self.class_num], np.float32) #bounding boxes
            ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
            #one-hot vectors with four classesimage1
            #[1,0,0,0] -> cat
            #[0,1,0,0] -> dog
            #[0,0,1,0] -> person
            #[0,0,0,1] -> background
            

            ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
            
            img_name = self.imgdir+self.img_names[index]
            
            
            #TODO:
            #1. prepare the image [3,320,320], by reading image "img_name" first.
            #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
            #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
            #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
    
            #to use function "match":
            #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
            #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
            
            #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
            #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        
            image = cv2.imread(img_name)
            
            width = image.shape[0]
            height = image.shape[1]
            # resize
            image = cv2.resize(image, (self.image_size,self.image_size))
            
            image = np.swapaxes(image,1,2) 
            image = np.swapaxes(image,0,1) # [3,320,320]
            
            if self.anndir == None:
                return image, ann_box, ann_confidence
            
            ann_name = self.anndir+self.img_names[index][:-3]+"txt"

            file_name = open(ann_name, "r")
            line = file_name.readlines()
            file_name.close()       
            for ln in line[0:len(line)]:
                line=ln.strip().split()
            
            class_id = int(line[0])
            w = float(line[3])
            h = float(line[4])
            x_c = float(line[1]) + w/2 
            y_c = float(line[2]) + h/2
            
            x_min = (x_c - w/2)/height
            y_min = (y_c - h/2)/width
            x_max = (x_c + w/2)/height
            y_max = (y_c + w/2)/width
            
            ann_box, ann_confidence = match(ann_box,ann_confidence,
                                            self.boxs_default,self.threshold,
                                            class_id,
                                            x_min,y_min,x_max,y_max)
            
            return image, ann_box, ann_confidence
        
        else: #data augmentation
            index = index-img_length
            
            ann_box = np.zeros([self.box_num,self.class_num], np.float32) #bounding boxes
            ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
            
            ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
            
            img_name = self.imgdir+self.img_names[index]
            ann_name = self.anndir+self.img_names[index][:-3]+"txt"
            
            image = cv2.imread(img_name)
            
            width = image.shape[0]
            height = image.shape[1]
            
            file_name = open(ann_name, "r")
            line = file_name.readlines()
            file_name.close()       
            for ln in line[0:len(line)]:
                line=ln.strip().split()
            
            class_id = int(line[0])
            # random crop around box
            w = float(line[3])
            h = float(line[4])
            x_c = float(line[1]) + w/2 
            y_c = float(line[2]) + h/2
            
            x_min = x_c - w/2
            y_min = y_c - h/2
            
            crop_x = random.randint(0, int(x_min))
            crop_y = random.randint(0, int(y_min))
            
            image = image[crop_y:-1,crop_x:-1,:]
            
            x_c = x_c - crop_x
            y_c = y_c - crop_y
            
            # resize
            image = cv2.resize(image, (self.image_size,self.image_size))
            image = cv2.flip(image, 1)
            
            image = np.swapaxes(image,1,2) 
            image = np.swapaxes(image,0,1) # [3,320,320]

            x_c = height - x_c
            
            x_min = (x_c - w/2)/height
            y_min = (y_c - h/2)/width
            x_max = (x_c + w/2)/height
            y_max = (y_c + w/2)/width
            
            ann_box, ann_confidence = match(ann_box,ann_confidence,
                                            self.boxs_default,self.threshold,
                                            class_id,
                                            x_min,y_min,x_max,y_max)
            
            return image, ann_box, ann_confidence

