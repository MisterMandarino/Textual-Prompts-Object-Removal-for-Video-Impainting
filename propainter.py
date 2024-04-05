import subprocess
import os
from PIL import Image
import numpy as np
import cv2
import torch

from utils.plot import plot_segmented_image, get_distance
from utils.video import display_video
from utils.process import runcmd

## Import the models
from ultralytics import YOLO    ## Object detection
import clip    ## Prompt Object Selection
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor    ## Object Segmentation

import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Load Prompts")
parser.add_argument('-d', '--DIR_PATH', type=str, required=True, help='Directory containing the video sequences')
parser.add_argument('-h_res', '--HIGH', type=bool, default=False, help='Set to True for higher resolution (more GPU memory requirement)')
parser.add_argument('-display', '--DISPLAY', type=bool, default=True, help='Set to True to display video results')

args = parser.parse_args()

## Get Working Directory
HOME = os.getcwd()

## Check user's parameters
if type(args.DIR_PATH) != str:
    parser.error("--DIR_PATH is required and must be a string")
else:
    DIRECTORY_PATH = os.path.join(HOME, args.DIR_PATH)
    if not os.path.exists(DIRECTORY_PATH):
        parser.error("--DIR_PATH does not exist!")
    MASK_PATH = DIRECTORY_PATH+'_mask'

PROMPT = input("Enter a Textual Prompt: ")
assert type(PROMPT) == str, 'The Prompt should be of type str'
HIGH = args.HIGH
DISPLAY = args.DISPLAY

## Get the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('Selected device: ', device)

## Download Segment Anything Model's weights
#sam_weight_path = os.path.join(HOME, 'sam_vit_h_4b8939.pth')
#if os.path.isfile(sam_weight_path):
#    print('SAM model\'s weights already downloaded!')
#else:
#    print('Downloading SAM\'s weights...')
#    runcmd("wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", verbose=True)

# Loading the Yolo model
print('Loading Yolo Model...')
yolo_model = YOLO("yolov8m.pt")

# Loading the clip model
print('Loading Clip Model...')
clip_model, preprocess = clip.load('RN50')  # Choosing RN50
clip_model.to(device).eval()

## Load the SAM model
print('Loading Sam Model...')
sam_weight_path = os.path.join(HOME, 'sam_vit_h_4b8939.pth')
model_type = 'vit_h'
sam_model = sam_model_registry[model_type](checkpoint=sam_weight_path).to(device=device)
sam_predictor = SamPredictor(sam_model)

## Create the video masks
if not os.path.exists(MASK_PATH):
    os.mkdir(MASK_PATH)
    print('Creating Video Mask Sequences...')
    first = True
    sorted_dir = sorted(os.listdir(DIRECTORY_PATH))
    for filename in sorted_dir:
        if filename.endswith('.jpg'):
            # Save path name
            PATH_NAME = os.path.join(DIRECTORY_PATH, filename)

            # get yolo's predictions
            results = yolo_model.predict(PATH_NAME)
            result = results[0]

            if first:
                image = Image.open(PATH_NAME)

                #Preprocess crops for clip image encoder
                batch_of_croppings = [image.crop(bbox.xyxy[0].tolist()) for bbox in result.boxes]
                batch_of_croppings = [preprocess(img).unsqueeze(0) for img in batch_of_croppings]
                batch_of_croppings = torch.stack(batch_of_croppings).squeeze(1).to(device)

                text_input = clip.tokenize(PROMPT).to(device)

                #Compute logits
                logits_per_image, logits_per_text = clip_model(batch_of_croppings, text_input)

                #Get the best bounding box
                best_bbox_index = torch.argmax(logits_per_image)
                predicted_bbox = result.boxes[best_bbox_index].xyxy[0].tolist()
                predicted_bbox = np.array(predicted_bbox)
                pivot = np.copy(predicted_bbox)
            else:
                max_distance = float('inf')
                for box in result.boxes:
                    if get_distance(pivot, np.array(box.xyxy[0].tolist())) < max_distance:
                        max_distance = get_distance(pivot, np.array(box.xyxy[0].tolist()))
                        predicted_bbox = np.array(box.xyxy[0].tolist())
                pivot = np.copy(predicted_bbox)

            ## initialize the sam predictor
            image_bgr = cv2.imread(PATH_NAME)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(image_rgb)

            # Get the segmentation mask
            masks, _, _ = sam_predictor.predict(box=predicted_bbox, multimask_output=False)
            mask = masks[0]

            if first:
                plot_segmented_image(image_bgr, masks)
                first = False

            ## Compute the BGR mask
            gray = mask.astype(np.uint8) * 255
            mask_bgr = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            mask_bgr[:, :, 0] = gray

            ## Save the mask
            mask_filename = os.path.join(MASK_PATH, filename.replace('.jpg', '.png'))
            cv2.imwrite(mask_filename, mask_bgr)
            print('Masks succesfully saved in ', MASK_PATH)
else:
    print('Video mask sequences already present!')

## ProPainter Inference
PROPAINTER_EXE = os.path.join(HOME, 'ProPainter', 'inference_propainter.py')
if HIGH:
    subprocess.run(["python", PROPAINTER_EXE, "--video", DIRECTORY_PATH, "--mask", MASK_PATH, "--height", str(320), "--width", str(576), "--fp16"])
else:
    subprocess.run(["python", PROPAINTER_EXE, "--video", DIRECTORY_PATH, "--mask", MASK_PATH])


## Define the video result path
PROPAINTER_PATH_IN = os.path.join(HOME, 'ProPainter', 'results', DIRECTORY_PATH, 'masked_in.mp4')
PROPAINTER_PATH_OUT = os.path.join(HOME, 'ProPainter', 'results', DIRECTORY_PATH, 'inpaint_out.mp4')
RESULT_PATH = os.path.join(HOME, 'results')
RESULT_PATH_IN = os.path.join(RESULT_PATH, DIRECTORY_PATH+'_in.mp4')
RESULT_PATH_OUT = os.path.join(RESULT_PATH, DIRECTORY_PATH+'_out.mp4')
if not os.path.isdir(RESULT_PATH):
    os.mkdir(RESULT_PATH)
os.rename(src=PROPAINTER_PATH_IN, dst=RESULT_PATH_IN)
os.rename(src=PROPAINTER_PATH_OUT, dst=RESULT_PATH_OUT)

## Display the videos
if DISPLAY:
    display_video(RESULT_PATH_IN, RESULT_PATH_OUT)