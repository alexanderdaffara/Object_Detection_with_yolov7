![header][]
Alexander Claudino Daffara
Flatiron DataScience Capstone Project
https://medium.com/@alexanderdaffara

Hi, I'm Alex, a Data Scientist, and in this project, I will be creating an object detection model with the aim of detecting vehicles and pedestrians. Here, I will provide a detailed overview of the project, including the business problem, data understanding, data preparation, metrics, and modeling approach.

# Repo Navigation
DataGathering_EDA.ipynb contains the data preprocessing and Exploration of images and labels  
Final_Notebook_Modelling.ipynb contains the modelling done with yolov7  
  
Unfortunately, the waymo data files are far too large to be uploaded to git.  
  
/data contains example images for inference  
/yolov7 contains files downloaded from the yolov7 package https://github.com/WongKinYiu/yolov7  


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
We started by comparing two pre-trained models, MobilenetSSD and Yolov7. The first simple model used MobileNetSSD with no custom training, which scored mAP .003. Yolov7 with no custom training scored mAP .005. Yolov7 trained on half the training data scored mAP .33, and Yolov7 trained on the full dataset scored mAP .27.

# Conclusion
The results of this project indicate that it is possible to use object detection models to estimate traffic volume at a vehicle's location using camera data. However, there is still room for improvement, especially in the development of more accurate models that can handle more complex scenarios. Future work could include the integration of lidar sensor data to further enhance the accuracy of our models.



