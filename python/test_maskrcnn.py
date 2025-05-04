import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load the model
model = maskrcnn_resnet50_fpn(num_classes=2)  # 1 class (tumor) + background
model.load_state_dict(torch.load('mask_tumor.pth'))
model.eval()  # Set the model to evaluation mode

# Load and preprocess the image
image_path = 'D:\Download\BrainTumor Classification DL\pred\pred45.jpg'  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
image_tensor = F.to_tensor(image)  # Convert to tensor
image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

# Make predictions
with torch.no_grad():
    predictions = model(image_tensor)

# Get the first prediction
pred = predictions[0]

# Visualize only the most confident prediction above a threshold
boxes = pred['boxes'].cpu().numpy()
scores = pred['scores'].cpu().numpy()
conf_thresh = 0.8
valid = scores > conf_thresh

if np.any(valid):
    best_idx = np.argmax(scores[valid])
    best_box = boxes[valid][best_idx].astype(int)
    box_color = (153, 159, 250)  
    cv2.rectangle(image, (best_box[0], best_box[1]), (best_box[2], best_box[3]), box_color, 2)
else:
    print("No tumor detected with confidence above threshold.")

# Display the image
cv2.imshow('Prediction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()