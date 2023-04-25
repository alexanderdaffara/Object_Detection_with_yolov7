![header](https://raw.githubusercontent.com/alexanderdaffara/Object_Detection_with_yolov7/main/data/15844593126368860820_3260_000_3280_000%3B1559178780337582.jpg)
Author: Alexander Claudino Daffara  
Flatiron DataScience Capstone Project  
LinkedIn: https://www.linkedin.com/in/alexanderdaffara/  
Blogs: https://medium.com/@alexanderdaffara  

Hi, I'm Alex, a Data Scientist, and in this project, I will be creating an object detection model with the aim of detecting vehicles and pedestrians. Here, I will provide a detailed overview of the project, including the business problem, data understanding, data preparation, metrics, and modeling approach.

# Repo Navigation
DataGathering_EDA.ipynb contains the data preprocessing and Exploration of images and labels  
Final_Notebook_Modelling.ipynb contains the modelling done with yolov7  
  
Unfortunately, the waymo data files are far too large to be uploaded to git.  
  
/data contains example images for inference  
/yolov7 contains utility scripts downloaded from the yolov7 package https://github.com/WongKinYiu/yolov7  


# Business Understanding
The problem we are addressing is that Google Maps wants to accurately estimate vehicle and pedestrian traffic volume in certain locations to better estimate optimal car commute routes. Mobility as a service (MaaS) is a growing market valued at $3.3 Billion in 2021 and expected to reach $40.1 Billion in 2030. Traffic monitoring systems at Google Maps for navigation systems use location-based speed monitoring and user inputted feedback on how much traffic they are experiencing at their location. With the increase in production and accessibility of autonomous vehicles and car-mounted dash cams, we have the demand and systems available to launch a tool that can use image and video data to estimate traffic volume at a vehicle's location.

# Data Understanding
The Waymo open perception dataset, which was made publicly available for their 2023 2D panoptic segmentation challenge, is used for this project. The Dataset contains millions of images and lidar sensor data from their self-driving vehicles and labelled bounding boxes (for vehicles, pedestrians, and cyclists). Since mass lidar installation on vehicles is expensive (around $1000 installation per vehicle), we will be trying to optimize our solution using camera data only. We are also using images only from the frontal-facing camera to optimize accuracy for easy installation dashcams. We are disregarding the cyclist target labels since analysis of cyclist labels was showing inconsistency (some fire hydrants and standing children were mistaken for cyclists). This leaves us with around 3 million object labels from 800 segments of contiguous 1920x1280 pixel camera data. These images are from urban roads and highways (consistent with our business problem).

# Data Preparation
We scale and letterbox our images to a size of 640x640 for consistent input for our model. This also allows the model to downscale high-resolution images for faster learning and predictions. No image data augmentation was necessary given the tremendous amount of data available.

# Metrics
The mean Average Precision (mAP) is a commonly used evaluation metric for object detection tasks. It evaluates how well a model is able to detect objects of interest in an image, by measuring the precision and recall of the model's predictions.

In object detection, a bounding box is considered a correct detection if it has an intersection over union (IoU) greater than a certain threshold with a ground truth bounding box. The IoU measures the overlap between the predicted and ground truth bounding boxes.

To calculate the mAP, the precision and recall values for each object class are first computed for different IoU thresholds. For example, at an IoU threshold of 0.5, a detection is considered a true positive if it has an IoU greater than 0.5 with a ground truth bounding box. At an IoU threshold of 0.6, a detection is considered a true positive if it has an IoU greater than 0.6 with a ground truth bounding box. The precision is then calculated as the ratio of true positives to the total number of predictions made by the model, while recall is calculated as the ratio of true positives to the total number of ground truth objects.

The precision-recall curve is then plotted for each object class by varying the IoU threshold. The mAP is calculated as the area under the precision-recall curve, averaged over all object classes. This measures the overall performance of the model in detecting objects across different IoU thresholds.

For your project, you will calculate the mAP for two object classes, vehicles and pedestrians, using the IoU threshold of 0.5 to 0.95. The mAP metric will help you evaluate and compare the performance of different models you train, and determine which one performs better in detecting vehicles and pedestrians in the images.

# Modelling
I am using the Yolov7 (You Only Look Once) model.  
Yolov7 with no custom training scored mAP 0.005.  
Yolov7 trained on half the training data scored mAP .33  

Yolov7 trained on half the training data with hyper-parameter tuning scored mAP .33  

Yolov7 trained on the full dataset scored mAP .27.

# Use case
Google Maps can use the following use case tool to determine the traffic level for a given street:
![usecase](https://raw.githubusercontent.com/alexanderdaffara/Object_Detection_with_yolov7/main/data/use_case.png)

# Conclusion
The developed software has the capability to be seamlessly integrated into various systems, including vehicles equipped with camera sensors, the Waymo autonomous driving platform, and existing low-cost vehicle dash cameras.

The tool is designed to provide additional information to improve the accuracy of real-time traffic volume estimation. By using image and video data to detect vehicles and pedestrians, the software can effectively enhance the existing traffic monitoring systems.

In particular, the integration of the software into Google Maps would help to improve the traffic cost function when recommending commute routes to users. With accurate traffic volume estimation, users can be provided with optimal routes based on real-time traffic conditions.

# Future work 
In the future I would:
1. Introduce training Data variety 
2. Gather data over time for specific locations so that I could estimate trends to determine traffic level and compare with current levels
3. Optimize the tool for parked cars and obstructive vehicles




