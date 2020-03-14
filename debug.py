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
#network.cuda()
cudnn.benchmark = True

#dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)    
dataset_test = COCO("data/test/images/", None, class_num, boxs_default, train = False, image_size=320)

dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
network.load_state_dict(torch.load('network_99.pth', map_location={'cuda:0': 'cpu'}))
network.eval()

#img_names = os.listdir("data/train/images/")
img_names = os.listdir("data/test/images/")

for i, data in enumerate(dataloader_test, 0):
    images_, ann_box_, ann_confidence_ = data
    images = images_#.cuda()
    ann_box = ann_box_#.cuda()
    ann_confidence = ann_confidence_#.cuda()

    pred_confidence, pred_box = network(images)

    # GPU
    #pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
    #pred_box_ = pred_box[0].detach().cpu().numpy()
    
    # CPU
    pred_confidence_ = pred_confidence[0].detach().numpy()
    pred_box_ = pred_box[0].detach().numpy()
    
    ##############################
    #result_image = visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
    ###############################
    #cv2.imwrite('visualized1/result_before_NMS_%d.jpg'%i, result_image)
    #cv2.waitKey(1000)
    
    
    #############################
    pred_confidence_,pred_box_, index = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)

    ann_confidence = ann_confidence_[0].numpy()
    ann_box = ann_box_[0].numpy()

    result_image = visualize_pred_NMS("test", pred_confidence_, pred_box_, 
                                      ann_confidence, ann_box, images_[0].numpy(), 
                                      boxs_default, index)
    ##############################
        
    #cv2.imwrite('visualized2/result_after_NMS_%d.jpg'%i, result_image)
    #cv2.waitKey(1000)
    
        #TODO: save predicted bounding boxes and classes to a txt file.
    #you will need to submit those files for grading this assignment
    # if img_names == os.listdir("data/test/images/"):
    
    ########################### save text file ########################
    ann_path = "predicted_boxes/"
    img_path = "data/test/images/"
    # get image name
    ann_name = img_names[i][:-4]
    img_name = img_names[i]
    save_ann_txt(ann_path, ann_name, img_path, img_name, pred_confidence_, pred_box_, boxs_default,index)
    print("saved image%d"%i)
    
    