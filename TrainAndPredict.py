import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import datetime
import time
import os
import gdal

from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from projects.PointRend.point_rend import add_pointrend_config

#获取tif图像仿射矩阵信息
def readTif(fileName):
 dataset = gdal.Open(fileName)
 if dataset == None:
   print(fileName+"文件无法打开")
 im_geotrans = dataset.GetGeoTransform()#获取仿射矩阵信息
 del dataset
 return im_geotrans
#写入仿射矩阵信息
def writeTif(fileName,im_geotrans):
 dataset = gdal.Open(fileName,gdal.GA_Update)
 if dataset == None:
    print(fileName + "文件无法打开")
 dataset.SetGeoTransform(im_geotrans)  # 获取仿射矩阵信息
 del dataset


def Train():
    register_coco_instances("custom", {}, "datasets/coco/annotations/instances_train2017.json", "datasets/coco/train2017")
    custom_metadata = MetadataCatalog.get("custom")
    dataset_dicts = DatasetCatalog.get("custom")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=custom_metadata, scale=1)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('Sample',vis.get_image()[:, :, ::-1])
        cv2.waitKey()


    cfg = get_cfg()
    cfg.merge_from_file(
        "configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("custom",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = 'model_final_3c3198.pkl'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
   
    cfg.SOLVER.MAX_ITER = (

        150000
    )  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    ) 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()    
 

def Predict():
    register_coco_instances("custom", {}, "datasets/coco/annotations/instances_train2017.json", "datasets/coco/train2017")
    custom_metadata = MetadataCatalog.get("custom")
    DatasetCatalog.get("custom")

    
    #im = cv2.imread("2.tif")

    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file("/home/user/Downloads/detectron2-master2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml")
#pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml
    cfg.DATASETS.TEST = ("custom", )
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0064999.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
      512
    )  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    file_root = '/home/user/Downloads/detectron2-master2/datasets/coco/test2017/'
    file_list = os.listdir(file_root)
    save_out = "./output/"

    for img_name in file_list:
        img_path = file_root + img_name
        im = cv2.imread(img_path)
        predictor = DefaultPredictor(cfg)
        img = np.zeros(im.shape, dtype=np.uint8)
        outputs = predictor(im)
        v = Visualizer(#im[:, :, ::-1],
                   img,
                   metadata=custom_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        save_path = save_out + img_name

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #v = v.draw_binary_mask(img)

        cv2.imshow('Result', v.get_image()[:, :, ::-1])
        #print(outputs["instances"].to("cpu"))

        cv2.imwrite(save_path,v.get_image()[:, :, ::-1])

        img1 = cv2.cvtColor(v.get_image()[:, :, ::-1],cv2.COLOR_RGB2GRAY)
        cv2.imwrite(save_out+ "gray_"+img_name ,img1)
        #im_proj = dataset.GetProjection()#获取投影信息
        w = readTif(img_path)
        print(w)
        writeTif(save_path,w)
        writeTif(save_out+"gray_"+img_name,w)

        cv2.waitKey(1)
     
if __name__ == "__main__":
    #Train()
    Predict()
