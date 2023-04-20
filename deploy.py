import streamlit as st
from PIL import Image
from pathlib import Path
import os
import torch
import cv2
import random
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box


def load_model():
    # load, build and return the model on cpu
    device = torch.device('cpu')
    model = torch.load(
        './model_weights/final_model_weights.pt', 
        map_location=device
    )['model'].float().fuse().eval()
    return model

def detect_and_save(model, source):
    # MAKE DETECTIONS, DRAW BOXES, SAVE TO FILE
    # define destination paths
    save_dir = Path('data/deploy_imgs/imgs_out')
    save_dir.mkdir(parents=True,exist_ok=True)

    # preprocess input image
    dataset = LoadImages(source)
    dataset.__iter__()
    path, img, img0, _ = dataset.__next__() # img0 is original image
    device = torch.device('cpu')
    img = torch.from_numpy(img).to(device) # shape: (3,384,640)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0) # shape: (1,3,384,640)

    # make detections
    with torch.no_grad():  
        pred = model(img)[0]
    # filter best detections
    pred = non_max_suppression(pred, conf_thres=.5) # list of shape (1,n_detec,6)

    # draw boxes and labels on image and save to file
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in model.names]
    for i, det in enumerate(pred):  # detections per image
        p = Path(path)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=4)
        cv2.imwrite(save_path, img0)

def save_uploadedfile(uploadedfile):
     # save user uploaded image file to input folder
     path_name = os.path.join("data/deploy_imgs/imgs_in",uploadedfile.name)
     with open(path_name,"wb") as f:
         f.write(uploadedfile.getbuffer())
     return path_name
# Load the model
model = load_model()


# Define the classes
classes = ['Cardboard', 'Carton', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Define the Streamlit app
def main():
    st.title("Vehicle & Pedestrian Detection")
    st.write("Upload an image to find the Vehicles and Pedestrians.")

    # Add file uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Make a prediction on the image
    if uploaded_file is not None:
        input_path = save_uploadedfile(uploaded_file)
        detect_and_save(model,input_path)

        save_dir = Path('data/deploy_imgs/imgs_out')
        save_dir.mkdir(parents=True,exist_ok=True)
        out_path = str(save_dir / uploaded_file.name)
        out_image = Image.open(out_path)
        st.image(out_image, caption='Detections', use_column_width=True)
        

# Run the app
if __name__ == '__main__':
    main()