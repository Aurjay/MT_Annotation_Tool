import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

HOME = os.getcwd()
print("HOME:", HOME)

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

if torch.cuda.is_available():
    # CUDA is available, use GPU
    DEVICE = torch.device('cuda:0')
    print("Using GPU:", torch.cuda.get_device_name(DEVICE))
else:
    # CUDA is not available, use CPU
    DEVICE = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_predictor = SamPredictor(sam)

DATA_DIR = r"Path to input folder"
output_dir = r"Path to output folder"
os.makedirs(output_dir, exist_ok=True)

# Initialize counter
counter = 1

for filename in os.listdir(DATA_DIR):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  
        print("Processing image:", filename)
        IMAGE_PATH = os.path.join(DATA_DIR, filename)

        # Load the image
        image = cv2.imread(IMAGE_PATH)

        expanded_image = torch.unsqueeze(torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255, dim=0)

        # Create a copy of the image for drawing bounding boxes
        image_with_boxes = image.copy()

        bounding_boxes = []

        drawing = False

        # Function to handle mouse events
        def draw_bbox(event, x, y, flags, param):
            global image_with_boxes
            global bounding_boxes
            global drawing

            if event == cv2.EVENT_LBUTTONDOWN:
                # Start drawing bounding box
                bounding_boxes.append([(x, y)])
                drawing = True

            elif event == cv2.EVENT_LBUTTONUP:
                bounding_boxes[-1].append((x, y))
                drawing = False

                cv2.rectangle(image_with_boxes, bounding_boxes[-1][0], bounding_boxes[-1][1], (0, 0, 255), 2)  # Red color
                cv2.imshow("Bounding Box", image_with_boxes)

        cv2.namedWindow("Bounding Box", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Bounding Box", 600, 600)  

        # Set mouse event callback
        cv2.setMouseCallback("Bounding Box", draw_bbox)

        # Display the image
        cv2.imshow("Bounding Box", image_with_boxes)

        # Wait until 's' key is pressed to submit bounding boxes for segmentation
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                # Clear bounding boxes and reset image
                bounding_boxes = []
                image_with_boxes = image.copy()
                cv2.imshow("Bounding Box", image_with_boxes)

            if key == ord("s"):
                if not bounding_boxes:  # If no bounding box drawn
                    # Create a black image
                    black_image = np.zeros_like(image)
                    output_filename = os.path.join(output_dir, f"ann-img{counter:04d}_gt.jpg")
                    cv2.imwrite(output_filename, black_image)
                    print(f"Blank image saved successfully as {output_filename}!")
                    counter += 1
                    break

                else:
                    aggregated_mask = np.zeros_like(image[:, :, 0])

                    for bbox in bounding_boxes:
                        bbox = np.array(bbox)
                        mask_predictor.set_image(image)
                        masks, _, _ = mask_predictor.predict(
                            box=bbox,
                            multimask_output=True
                        )

                        for mask in masks:
                            aggregated_mask[mask > 0] = 255

                    # Display the aggregated mask
                    cv2.imshow("Aggregated Mask", aggregated_mask)

                    # 's' to save, 'c' to clear bounding boxes and redraw
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("s"):
                        output_filename = os.path.join(output_dir, f"ann-img{counter:04d}_gt.jpg")
                        cv2.imwrite(output_filename, aggregated_mask)
                        print(f"Aggregated mask saved successfully as {output_filename}!")
                        counter += 1
                        break
                    elif key == ord("c"):
                        bounding_boxes = []
                        image_with_boxes = image.copy()
                        cv2.imshow("Bounding Box", image_with_boxes)

        cv2.destroyAllWindows()
