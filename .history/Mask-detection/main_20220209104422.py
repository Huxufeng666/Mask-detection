from turtle import Turtle
from webbrowser import Elinks
from numpy import source
import torch
import numpy
from torch import nn
import torch.nn.functional as F
import sys
import cv2
from yolov4.models import Yolov4
from yolov4.tool.torch_utils import *
from yolov4.tool.yolo_layer import YoloLayer
from yolov5.detect import *
from yolov5.utils.general import *
from yolov4.tool.utils import load_class_names, plot_boxes_cv2
from yolov4.tool.torch_utils import do_detect, get_region_boxes
import argparse
import os
from pathlib import Path
import torch.backends.cudnn as cudnn

from ensemble_boxes import *


'''
yolov4
'''
namesfile = 'Mask-detection/yolov4/data/Mask.names'
n_classes = 3
weights_v4 = 'Mask-detection/yolov4/Yolov4_Mask_detection.pth'
weights_v5 = 'Mask-detection/yolov5/runs/train/exp9/weights/last.pt'
source = 'Mask-detection/image/56.jpg'
imgsz = (608,608)
model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
pretrained_dict = torch.load(weights_v4, map_location=torch.device('cuda'))
model.load_state_dict(pretrained_dict)
use_cuda = True
device = ''

if use_cuda:
    model.cuda()
img = cv2.imread(source)
sized = cv2.resize(img, imgsz)
sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

'''
yolov5

'''
def NMS(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]#取出det中的X1
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # 按置信度的大小降序排列

    keep = []  # 需要保留的bounding box
    while order.size > 0:
        i = order[0]  # 取置信度最大的框 NMS的原则是det中，取最大的置信度最大的一个先保留。剩下的分别计算与最大置信度框的交并比，若交并比大于阈值，则判定该框与最大置信度框的目标为同一目标，删去该框。之后选择置信度第二高的框，再执行上面的过程。
        keep.append(i)  # 将其作为保留的框

        
        xx1 = np.maximum(x1[i], x1[order[1:]])  # 计算xmin的max,即overlap的xmin 将置信度最大的框与其他框的坐标相比，若最大框的坐标在右，则用其坐标替换，目的是求出两框的交集。
        yy1 = np.maximum(y1[i], y1[order[1:]])  # 计算ymin的max,即overlap的ymin
        xx2 = np.minimum(x2[i], x2[order[1:]])  # 计算xmax的min,即overlap的xmax
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 计算ymax的min,即overlap的ymax

        w = np.maximum(0.0, xx2 - xx1 + 1)  # 计算overlap的width
        h = np.maximum(0.0, yy2 - yy1 + 1)  # 计算overlap的hight
        inter = w * h  # 计算overlap的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # 计算并，-inter是因为交集部分加了两次。

        inds = np.where(ovr <= thresh)[0]  # 若ovr中的数没超过阈值，则保存ovr中该数的下标至inds
        order = order[inds + 1]  # 删除IOU大于阈值的框

    return keep



def xywhn2xyxy(x, w=608, h=608, padw=0, padh=0):

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 2] - x[:, 0] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 3] - x[:, 1] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 2] + x[:, 0] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 3] + x[:, 1] / 2) + padh  # bottom right y
    return y



if __name__ == "__main__":
   

# yolov4 bbox

    for i in range(2):  # This 'for' loop is for speed check
                        # Because the first iteration is usually longer
        bbox = do_detect(model, sized, 0.4, 0.6, use_cuda)

    class_names = load_class_names(namesfile)
    boxes =  plot_boxes_cv2(img, bbox[0], 'runs/56.jpg',None)


    
    bbox_v4 = np.array(boxes)
    # bbox_v4 = np.array(bbox)
    # b_4=[i[:6] for item in bbox_v4 for i in item]
    # s_4=[i[4:5] for item in bbox_v4 for i in item] 
    # l_4=[i[5:6] for item in bbox_v4 for i in item] 



    # print(s_4)
    
# yolov5 bbox

    bbox = run(weights_v5=weights_v5,  # model.pt path(s)
        source=source,  # file/dir/URL/glob, 0 for webcam
        imgsz=imgsz,  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/',  # save results to project/name
        name='yolov5',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        )
    bbox_v5 = bbox.cpu().detach().numpy()
    # bbox_v5 =np.array(bbox)
    # b_5=[i[:6] for item in bbox_v5 for i in item]
    # s_5=[i[4:5] for item in bbox_v5 for i in item] 
    # l_5=[i[5:6] for item in bbox_v5 for i in item] 

    print(bbox_v5)
    



# merged

    # bbox_merged_2 =  np.append(b_4, b_5, axis= 0)   
    # boxes_list= np.append(b_4, b_5, axis= 0)
    # scores_list=np.append(s_4, s_5, axis= 0) 
    # labels_list=np.append(l_4, l_5, axis= 0) 

    bbox_merged_2 =  np.append(bbox_v5,bbox_v4, axis=0)   
  
    # b_v4=[]
    # b_v5=[]
    # bbox_merged = [] 
    # for x in torch.from_numpy(bbox_v4):
       
    #     b_v4=x[0],x[1],x[2],x[3]
     
    #     mul_v4 = (x[2]-x[0]) * (x[3]-x[1])  

    #     for y in torch.from_numpy(bbox_v5):

    #         b_v5=y[0],y[1],y[2],y[3]  
        
    #         mul_v5 = (y[2]-y[0]) * (y[3]-y[1])

    #     if mul_v4 > mul_v5:
    #         bbox_merged.append([y[0],y[1],y[2],y[3]])
    #     else:
    #         bbox_merged.append([x[0],x[1],x[2],x[3]])
        
            # if len(bbox_v5)> len(bbox_v4):
    #    bbox_merged=bbox_v5 
    
    # # elif mul_v4 > mul_v5:
    # #    bbox_merged=bbox_v5
    # else:
    #     bbox_merged=bbox_v4
    
    
    # bbox_merged = [] 
    # for x in bbox_merged_2:
    #     # if not x in bbox_merged :
    #     if  x.all() != bbox_merged.all() :

    #         bbox_merged.append(x)
 

    print(bbox_merged_2)


# ensenle_boxes


                                            

# boxe = NMS(boxes_list,0.8)
# boxes = boxes_list[boxe]


# print(boxes)


# box to image

# img = cv2.imread(str(source))
# h, w = img.shape[:2]

# # labels_list[:,4] = xywhn2xyxy(labels_list[:,4], w, h, 0, 0)

# for _, x in enumerate(bbox_merged):

#     cv2.rectangle(img,(int(x[0]),int(x[1])),(int(x[2]),int(x[3])),(255,0,0),2 )
#     cv2.putText(img,None,(int(x[0]), int(x[1] - 2)),fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(255,0, 0),thickness=1)

# cv2.imwrite('./text6.jpg',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
