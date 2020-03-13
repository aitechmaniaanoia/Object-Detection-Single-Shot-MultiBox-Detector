import os
import numpy as np
import time
import cv2

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

from dataset import *
from model import *
from utils import *

class_num = 4 # cat dog person background

#num_epochs = 100 #100
batch_size = 16

boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])

#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True

#dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)    
dataset_test = COCO("data2/test/images/", None, class_num, boxs_default, train = False, image_size=320)

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
network.load_state_dict(torch.load('network_39.pth'))
network.eval()

#img_names = os.listdir("data/train/images/")
img_names = os.listdir("data2/test/images/")

for i, data in enumerate(dataloader_test, 0):
    images_, ann_box_, ann_confidence_ = data
    images = images_.cuda()
    ann_box = ann_box_.cuda()
    ann_confidence = ann_confidence_.cuda()

    pred_confidence, pred_box = network(images)

    pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
    pred_box_ = pred_box[0].detach().cpu().numpy()
    
    
    #############################
    pred_confidence_,pred_box_, index = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)

    ann_confidence = ann_confidence_[0].numpy()
    ann_box = ann_box_[0].numpy()
    
    #ann_confidence_input = ann_confidence[index,:].reshape((len(index),4))
    #ann_box_input = ann_box[index,:].reshape((len(index),4))
    
    #boxs_default_input = boxs_default[index,:].reshape((len(index),8))

    # result_image = visualize_pred_NMS("test", pred_confidence_, pred_box_, 
    #                                   ann_confidence, ann_box, images_[0].numpy(), 
    #                                   boxs_default, index)
    ##############################
    
    
    #result_image = visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
    
    #cv2.imwrite('visualized2/result_after_NMS_%d.jpg'%i, result_image)
    #cv2.waitKey(1000)
    
        #TODO: save predicted bounding boxes and classes to a txt file.
    #you will need to submit those files for grading this assignment
    # if img_names == os.listdir("data/test/images/"):
    
    ann_path = "data2/test/annotations/"
    # get image name
    ann_name = img_names[i][:-4]
    save_ann_txt(ann_path, ann_name, pred_confidence_, pred_box_)
    
    
    