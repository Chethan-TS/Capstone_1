## Capstone_1

Manually labeling video data for object detection, especially in aerial footage, is time-consuming and impractical at scale. This project proposes a hybrid approach that combines traditional background subtraction (BGS) techniques with semi-supervised learning to automate the annotation of moving objects—primarily vehicles—with minimal manual effort. 

Four BGS methods—Frame Differencing, Running Average, GMM, and KNN—are used to detect foreground objects. Each method handles different motion and lighting conditions, and their outputs are fused to generate accurate segmentation masks. These masks are then converted into YOLO format to form a set of pseudo-labels.

To train a robust YOLOv8 object detector, only 10% of the dataset is manually annotated, while the rest is automatically labeled using model predictions. Two fusion strategies—vote-based and weighted—are implemented and compared to evaluate their impact on model performance.

This pipeline significantly reduces manual annotation workload and is ideal for traffic analysis, aerial surveillance, and other applications where labeled video data is scarce.

#Keywords: Deep Learning, Computer Vision, Background Subtraction,Semi-Supervised Learning, Object Detection, Object tracking
