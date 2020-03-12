import numpy as np
import cv2
from dataset import iou
import math


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use red green blue to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    
    img_size = image.shape[0]
    
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                #you can use cv2.rectangle as follows:
                gx = boxs_default[i,2]*ann_box[i,0] + boxs_default[i,0] # p default   d ann_box
                gy = boxs_default[i,3]*ann_box[i,1] + boxs_default[i,1]
                gw = boxs_default[i,2]*math.exp(ann_box[i,2])
                gh = boxs_default[i,3]*math.exp(ann_box[i,3])
                
                x1 = int((gx - gw/2)*img_size)
                y1 = int((gy - gh/2)*img_size)
                x2 = int((gx + gw/2)*img_size)
                y2 = int((gy + gh/2)*img_size)
                
                #print(x1,y1,x2,y2)
                
                start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                end_point = (x2, y2) #bottom right corner
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                image1 = cv2.rectangle(image1, start_point, end_point, color, thickness)
                
                ## draw ground truth "default" boxes on image2
                start_pt = (int(boxs_default[i,4]*img_size), int(boxs_default[i,5]*img_size))
                end_pt = (int(boxs_default[i,6]*img_size), int(boxs_default[i,7]*img_size))
                
                image2 = cv2.rectangle(image2, start_pt, end_pt, color, thickness)
    #aa = 0
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #aa = aa + 1
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                gx = boxs_default[i,2]*pred_box[i,0] + boxs_default[i,0] # p default   d ann_box
                gy = boxs_default[i,3]*pred_box[i,1] + boxs_default[i,1]
                gw = boxs_default[i,2]*math.exp(pred_box[i,2])
                gh = boxs_default[i,3]*math.exp(pred_box[i,3])
                
                x1 = int((gx - gw/2)*img_size)
                y1 = int((gy - gh/2)*img_size)
                x2 = int((gx + gw/2)*img_size)
                y2 = int((gy + gh/2)*img_size)
                
                #print(x1,y1,x2,y2)
                
                start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                end_point = (x2, y2) #bottom right corner
                color = colors[j] #use red green blue to represent different classes
                thickness = 2
                
                image3 = cv2.rectangle(image3, start_point, end_point, color, thickness)

                #draw network-predicted "default" boxes on image4
                start_pt = (int(boxs_default[i,4]*img_size), int(boxs_default[i,5]*img_size))
                end_pt = (int(boxs_default[i,6]*img_size), int(boxs_default[i,7]*img_size))
                
                image4 = cv2.rectangle(image4, start_pt, end_pt, color, thickness)
    #print(aa)
                
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(1)
    
    if windowname == 'test':
        # save image
        return image
    #return image
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    pred_box = []
    pred_confidence = []
    
    for k in range(confidence_.shape[1]): # for each classs
        # calculate max confidence
        if len(confidence_[:,k]) == 0:
            break
        else:
            max_confidence = max(confidence_[:,k])
        while max_confidence > threshold:
            # get box with max confidence
            idx = np.where(confidence_[:,k] == max_confidence)
            box_max_conf = box_[idx,:]
            confidence_max_conf = confidence_[idx,:]
            
            # remove max from box
            box_ = np.delete(box_, idx, axis=0)
            confidence_ = np.delete(confidence_, idx, axis=0)
            boxs_default = np.delete(boxs_default, idx, axis=0)
            
            # calculate ious for others
            x_min = box_[:,0] - box_[:,2]/2
            y_min = box_[:,1] - box_[:,3]/2
            x_max = box_[:,0] + box_[:,2]/2
            y_max = box_[:,1] + box_[:,3]/2
        
            ious = iou(boxs_default[:,:],x_min,y_min,x_max,y_max)
            idx_ = np.where(ious > overlap)
            
            #remove box with large iou
            box_ = np.delete(box_, idx_, axis=0)
            confidence_ = np.delete(confidence_, idx_, axis=0)
            boxs_default = np.delete(boxs_default, idx_, axis=0)
            
            # calculate new max
            if len(confidence_[:,k])==0:
                break
            else:
                max_confidence = max(confidence_[:,k])
            
            pred_confidence.append(confidence_max_conf)
            pred_box.append(box_max_conf)
    
    pred_box = np.array(pred_box)
    pred_confidence = np.array(pred_confidence)
    
    pred_box = pred_box.reshape((pred_box.shape[0], pred_box.shape[-1]))
    pred_confidence = pred_confidence.reshape((pred_confidence.shape[0], pred_confidence.shape[-1]))
            
    return pred_confidence, pred_box

    # #TODO: non maximum suppression
    # final_bounding_box = []
    # final_confidence = []
    # classes = []
    
    # for k in range(confidence_.shape[1]):   # for each classes
    #     #box = box_[:,:]
    #     #prob = confidence_[:,k]
    #     #C = boxs_default[:,:]
    #     #B = []
    #     #C = []
        
    #     #max_prob = box_[max(confidence_[:,k])]
    #     ##GET INDEX OF MAX_PROB
    #     idx = np.where(confidence_[:,k] == max(confidence_[:,k]))
    #     max_prob = box_[idx,:]
    #     max_prob_confidence = confidence_[idx,:]
    #     if max(confidence_[:,k]) > threshold:
    #         #B.append(max_prob)
    #         #C.append(max_prob_confidence)
    #         # remove max_prob from A
    #         box_ = np.delete(box_, idx, axis=0)
    #         confidence_ = np.delete(confidence_, idx, axis=0)
    #         boxs_default = np.delete(boxs_default, idx, axis=0)
            
    #         # for all boxes in A
    #         x_min = box_[:,0] - box_[:,2]/2
    #         y_min = box_[:,1] - box_[:,3]/2
    #         x_max = box_[:,0] + box_[:,2]/2
    #         y_max = box_[:,1] + box_[:,3]/2
        
    #         ious = iou(boxs_default[:,:],x_min,y_min,x_max,y_max)
        
    #         #ious_large = ious > overlap
    #         idx_ = np.where(ious > overlap)
        
    #         # remove large from A
    #         box = np.delete(box_, idx_, axis=0)
    #         confidence = np.delete(confidence_, idx_, axis=0)
    #         boxs_default_ = np.delete(boxs_default, idx_, axis=0)

    #         #B_all.append(B) # final boxes
    #         max_prob = max_prob.reshape((1,4))
    #         max_prob_confidence = max_prob_confidence.reshape((1,4))
    #         final_bounding_box.append(max_prob)
    #         final_confidence.append(max_prob_confidence)
    #         classes.append(k)
        
    #     pred_box_ = np.array(final_bounding_box)
    #     pred_confidence_ = np.array(final_confidence)
        
    #     pred_box_ = pred_box_.reshape((pred_box_.shape[0], pred_box_.shape[-1]))
    #     pred_confidence_ = pred_confidence_.reshape((pred_confidence_.shape[0], pred_confidence_.shape[-1]))
    
    # return pred_confidence_, pred_box_

def save_ann_txt(ann_path, ann_name, confidence, box):
    # class id when confidence max
    class_id = 0
    w = box[2]
    h = box[3]
    x = box[0] - w/2
    y = box[1] - h/2                                                                                                                                                                                [0] 
    line = [class_id, x, y, w, h]
    
    with open(ann_path + "%s.txt" %ann_name, "w") as text_file:
        text_file.write(line)
    








