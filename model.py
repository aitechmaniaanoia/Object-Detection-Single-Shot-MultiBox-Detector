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
    idx = torch.where(ann_confidence[:,-1] == 0)
    idx_empty = torch.where(ann_confidence[:,-1] == 1)
    
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
        self.Seq_conv = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,stride=2, padding=1),
                                      nn.BatchNorm2d(64), nn.ReLU(),
                                      
                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(64), nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(64), nn.ReLU(),
                                      
                                      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(128), nn.ReLU(),
                                      
                                      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(128), nn.ReLU(),
                                      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(128), nn.ReLU(),
                                      
                                      nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(256), nn.ReLU(),
                                      
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256), nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256), nn.ReLU(),
                                      
                                      nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(512), nn.ReLU(),
                                      
                                      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(512), nn.ReLU(),
                                      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(512), nn.ReLU(),
                                      
                                      nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(256), nn.ReLU()
                                      )
        
        ## second part: split two ways
        # left
        self.Seq_left_1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(256), nn.ReLU(),
                                      
                                      nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(256), nn.ReLU()
                                      )
        
        self.Seq_left_2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(256), nn.ReLU(),
                                      
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1),
                                      nn.BatchNorm2d(256), nn.ReLU()
                                      )
        
        self.Seq_left_3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, stride=1),
                                      nn.BatchNorm2d(256), nn.ReLU(),
                                      
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1),
                                      nn.BatchNorm2d(256), nn.ReLU()
                                      )
        
        self.Seq_left_4_box = nn.Conv2d(256, 16, kernel_size=1, stride=1, bias=True)
        self.Seq_left_4_confidence = nn.Conv2d(256, 16, kernel_size=1, stride=1, bias=True)
        
        # right
        self.right_1_box = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.right_1_confidence = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.right_2_box = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.right_2_confidence = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.right_3_box = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.right_3_confidence = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)
        
        #self.conv_256_16_3_1 = nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1, bias=True)
        
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
        
        x = self.Seq_conv(x)
        
        #second part
        # left 
        x_l_1 = self.Seq_left_1(x)
        x_l_2 = self.Seq_left_2(x_l_1)
        
        x_l_3_box = self.Seq_left_3(x_l_2)
        x_l_3_confidence = self.Seq_left_3(x_l_2)
        
        x_l_final_box = self.Seq_left_4_box(x_l_3_box)
        x_l_final_confidence = self.Seq_left_4_confidence(x_l_3_confidence)
        
        x_l_final_box = x_l_final_box.reshape((batch_size, 16, 1)) # [N,16,1]
        x_l_final_confidence = x_l_final_confidence.reshape((batch_size, 16, 1)) # [N,16,1]
        
        # right 
        x_r_box = self.right_1_box(x)
        x_r_confidence = self.right_1_confidence(x)

        x_r_box = x_r_box.reshape((batch_size, 16, 100)) #[N,16,100]        

        x_r_confidence = x_r_confidence.reshape((batch_size, 16, 100)) #[N,16,100]
        
        x_r_1_box = self.right_2_box(x_l_1)

        x_r_1_box = x_r_1_box.reshape((batch_size, 16, 25)) #[N.16.25]
        x_r_1_confidence = self.right_2_confidence(x_l_1)
        x_r_1_confidence = x_r_1_confidence.reshape((batch_size, 16, 25)) #[N.16.25]
        
        x_r_2_box = self.right_3_box(x_l_2)

        x_r_2_box = x_r_2_box.reshape((batch_size, 16, 9))
        x_r_2_confidence = self.right_3_confidence(x_l_2)
        x_r_2_confidence = x_r_2_confidence.reshape((batch_size, 16, 9))
        
        # concatenate
        bboxes = torch.cat((x_r_box, x_r_1_box, x_r_2_box, x_l_final_box,), axis = 2) 
        bboxes = bboxes.permute(0, 2, 1)
        bboxes = bboxes.reshape((batch_size, 540, 4))
        
        # confidence (box + softmax)
        confidence = torch.cat((x_r_confidence, x_r_1_confidence, x_r_2_confidence, x_l_final_confidence), axis = 2) 
        confidence = confidence.permute(0, 2, 1)
        confidence = confidence.reshape((batch_size, 540, 4))
        confidence = torch.softmax(confidence, dim = 2)
        
        return confidence,bboxes










