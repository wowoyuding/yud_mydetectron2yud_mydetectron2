**pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml**

python train_net.py --config-file configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml --num-gpus 2 SOLVER.IMS_PER_BATCH 4 MODEL.WEIGHTS /home/user/Downloads/detectron2-master2/model_final_ba17b9.pkl

