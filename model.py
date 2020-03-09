import os
import random
import numpy as np

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




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.binary_cross_entropy)
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.
    
    batch_size = pred_confidence.shape[0]
    num_boxes = pred_confidence.shape[1]
    num_classes = pred_confidence.shape[2]
    
    # reshape
    pred_confidence = pred_confidence.reshape((batch_size*num_boxes, num_classes))
    ann_confidence = ann_confidence.reshape((batch_size*num_boxes, num_classes))
    
    pred_box = pred_box.reshape((batch_size*num_boxes, 4))
    ann_box = ann_box.reshape((batch_size*num_boxes, 4))
    
    # before_reshape[i,j,k] = after_reshape[i*num_boxes+j,k]
    
    # get indices of all cells carrying objects in ground truth
    idx = torch.where(ann_confidence == 1)
    idx_empty = torch.where(ann_confidence == 0)
    
    # confidence  
    err_confidence = F.binary_cross_entropy(pred_confidence[idx], ann_confidence[idx]) + 3*F.binary_cross_entropy(pred_confidence[idx_empty], ann_confidence[idx_empty])
    
    # box
    err_box = F.smooth_l1_loss(pred_box[idx], ann_box[idx])
    
    err = err_confidence + err_box
    
    return err
    


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True) # 2 times
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True) # 2 times
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True)
        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True) # 2 times
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True)
        
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True) # 2 times
        self.conv9 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, bias=True)
        
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        
        ## second part: split two ways
        # left
        self.conv_256_256_1_1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=True)
        self.conv_256_256_3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv_256_256_3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, bias=True) 
        
        self.conv_256_16_1_1 = nn.Conv2d(256, 16, kernel_size=1, stride=1, bias=True)
        
        # right
        self.conv_256_16_3_1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)
        
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        
        #remember to apply softmax to confidence! Which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        batch_size = len(x)
        
        x = self.conv1(x)
        x = F.relu(self.bn64(x))
        
        x = self.conv2(x)
        x = F.relu(self.bn64(x))
        x = self.conv2(x)
        x = F.relu(self.bn64(x))
        
        x = self.conv3(x)
        x = F.relu(self.bn128(x))
        
        x = self.conv4(x)
        x = F.relu(self.bn128(x))
        x = self.conv4(x)
        x = F.relu(self.bn128(x))
        
        x = self.conv5(x)
        x = F.relu(self.bn256(x))
        
        x = self.conv6(x)
        x = F.relu(self.bn256(x))
        x = self.conv6(x)
        x = F.relu(self.bn256(x))
        
        x = self.conv7(x)
        x = F.relu(self.bn512(x))
        
        x = self.conv8(x)
        x = F.relu(self.bn512(x))
        x = self.conv8(x)
        x = F.relu(self.bn512(x))
        
        x = self.conv9(x)
        x = F.relu(self.bn256(x))
        
        #second part
        # left 
        x_l = self.conv_256_256_1_1(x)
        x_l = F.relu(self.bn256(x_l))
        x_l = self.conv_256_256_3_2(x_l)
        x_l = F.relu(self.bn256(x_l)) # [N,256,5,5]
        
        x_l_1 = self.conv_256_256_1_1(x_l)      
        x_l_1 = F.relu(self.bn256(x_l_1))
        x_l_1 = self.conv_256_256_3_1(x_l_1) 
        x_l_1 = F.relu(self.bn256(x_l_1)) #[N,256,3,3]
        
        x_l_2 = self.conv_256_256_1_1(x_l_1)
        x_l_2 = F.relu(self.bn256(x_l_2))
        x_l_2 = self.conv_256_256_3_1(x_l_2)
        x_l_2 = F.relu(self.bn256(x_l_2)) #[N,256,1,1]
        
        x_l_final = self.conv_256_16_1_1(x_l_2) # [N,16,1,1]
        # x_l_final reshape for box, confidence
        x_l_final = x_l_final.reshape((batch_size, 16, 1)) # [N,16,1]
        
        # right 
        x_r = self.conv_256_16_3_1(x)  #[N,16,10,10]
        x_r = x_r.reshape((batch_size, 16, 100)) #[N.16.100]
        
        x_r_1 = self.conv_256_16_3_1(x_l)
        x_r_1 = x_r_1.reshape((batch_size, 16, 25)) #[N.16.25]
        
        x_r_2 = self.conv_256_16_3_1(x_l_1)
        x_r_2 = x_r_2.reshape((batch_size, 16, 9))
        
        # concatenate
        #bboxes = np.concatenate((x_l_final, x_r, x_r_1, x_r_2), axis = 2)
        bboxes = torch.cat((x_l_final, x_r, x_r_1, x_r_2), axis = 2) 
        bboxes = bboxes.permute(0, 2, 1)
        bboxes = bboxes.reshape((batch_size, 540, 4))
        
        # confidence (box + softmax)
        confidence = F.softmax(bboxes)
        
        return confidence,bboxes










