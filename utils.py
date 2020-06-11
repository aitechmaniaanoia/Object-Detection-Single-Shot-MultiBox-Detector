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

    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
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



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.3, threshold=0.5):
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
    index = []
    pred_box = []
    pred_confidence = []
    
    gx = boxs_default[:,2]*box_[:,0] + boxs_default[:,0] # p default   d pred_box
    gy = boxs_default[:,3]*box_[:,1] + boxs_default[:,1]
    gw = boxs_default[:,2]*np.exp(box_[:,2])
    gh = boxs_default[:,3]*np.exp(box_[:,3])
    
    for k in range(confidence_.shape[1]-1): # for each classs except background
        # calculate max confidence
        if len(confidence_[:,k]) > 0:
            max_confidence = max(confidence_[:,k])
            
            while max_confidence > threshold:
                # get box with max confidence
                idx = np.where(confidence_[:,k] == max_confidence)
                idx = int(idx[0])
                index.append(idx)
                box_max_conf = box_[idx,:].copy()
                confidence_max_conf = confidence_[idx,:].copy()

                # calculate ious for others
                x_min = gx - gw/2
                y_min = gy - gh/2
                x_max = gx + gw/2
                y_max = gy + gh/2
                
                inter = np.maximum(np.minimum(x_max[idx],x_max)-np.maximum(x_min[idx],x_min),0)*np.maximum(np.minimum(y_max[idx],y_max)-np.maximum(y_min[idx],y_min),0)
                area_a = (x_max[idx]-x_min[idx])*(y_max[idx]-y_min[idx])
                area_b = (x_max-x_min)*(y_max-y_min)
                union = area_a + area_b - inter
                ious = inter/np.maximum(union,1e-8)

                idx_ = np.where(ious > overlap)
                idx_ = idx_[0]

                box_[idx_,:] = 0
                confidence_[idx_,:] = 0
            
                
                confidence_max_conf = confidence_max_conf.reshape((1,4))
                box_max_conf = box_max_conf.reshape((1,4))
                
                pred_confidence.append(confidence_max_conf)
                pred_box.append(box_max_conf)
                
                max_confidence = max(confidence_[:,k])
                
    pred_box = np.array(pred_box)
    pred_confidence = np.array(pred_confidence)
    
    pred_box = pred_box.reshape((pred_box.shape[0], pred_box.shape[-1]))
    pred_confidence = pred_confidence.reshape((pred_confidence.shape[0], pred_confidence.shape[-1]))
            
    return pred_confidence, pred_box, index

def visualize_pred_NMS(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default, index):
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

    #pred
    for i in range(len(pred_confidence)):
        classes = np.where(pred_confidence[i,:] == max(pred_confidence[i,:]))
        classes = int(classes[0])
        #print(classes)
        
        idx = index[i]  # index in default box
        #idx = int(idx[0])
        
        gx = boxs_default[idx,2]*pred_box[i,0] + boxs_default[idx,0] # p default   d ann_box
        gy = boxs_default[idx,3]*pred_box[i,1] + boxs_default[idx,1]
        gw = boxs_default[idx,2]*math.exp(pred_box[i,2])
        gh = boxs_default[idx,3]*math.exp(pred_box[i,3])
        
        x1 = int((gx - gw/2)*img_size)
        y1 = int((gy - gh/2)*img_size)
        x2 = int((gx + gw/2)*img_size)
        y2 = int((gy + gh/2)*img_size)
        
        start_point = (x1, y1) #top left corner, x1<x2, y1<y2
        end_point = (x2, y2) #bottom right corner
        color = colors[classes] #use red green blue to represent different classes
        thickness = 2
        
        image3 = cv2.rectangle(image3, start_point, end_point, color, thickness)

        #draw network-predicted "default" boxes on image4
        start_pt = (int(boxs_default[idx,4]*img_size), int(boxs_default[idx,5]*img_size))
        end_pt = (int(boxs_default[idx,6]*img_size), int(boxs_default[idx,7]*img_size))
        
        image4 = cv2.rectangle(image4, start_pt, end_pt, color, thickness)
                
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(1)
    #print("image_%d"%i)
    
    if windowname == 'test':
        # save image
        return image

def save_ann_txt(ann_path, ann_name, img_path, img_name,confidence, box, boxs_default, idx):
    if box.shape[0] > 0:       
        gx = boxs_default[idx,2]*box[:,0] + boxs_default[idx,0] # p default   d pred_box
        gy = boxs_default[idx,3]*box[:,1] + boxs_default[idx,1]
        gw = boxs_default[idx,2]*np.exp(box[:,2])
        gh = boxs_default[idx,3]*np.exp(box[:,3])
        
    img = cv2.imread(img_path + img_name)
    height = img.shape[0]
    width = img.shape[1]
    
    with open(ann_path + "%s.txt" %ann_name, "w") as text_file:
        if box.shape[0] == 0:
            text_file.write('\n')
        else:
            for i in range(box.shape[0]):
                class_id = np.where(confidence[i,:] == max(confidence[i,:]))
                class_id = int(class_id[0])

                line = [class_id, (gx[i] - gw[i]/2)*width, (gy[i] - gh[i]/2)*height, gw[i]*width, gh[i]*height]
                
                text_file.writelines(["%s," % item  for item in line[0:-1]])
                text_file.write("%s" % line[-1])
                text_file.write('\n')
    








