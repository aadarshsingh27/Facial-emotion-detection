import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import mediapipe as mp

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# -- Load Gender Model
gender_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
gender_model = torchvision.models.efficientnet_b0(weights=gender_weights).to(device)

for param in gender_model.parameters():
    param.requires_grad = True
    
gender_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 1)

).to(device)

# -- Load age Model
age_weights = torchvision.models.VGG19_Weights.DEFAULT
age_model = torchvision.models.vgg19( weights=age_weights)

# Freeze all parameters in the feature layers
for param in age_model.features.parameters():
    param.requires_grad = False

# Unfreeze only the last 20 layers of the feature layers
for param in age_model.features[-20:].parameters():
    param.requires_grad = True


age_model.classifier = nn.Sequential(
    
    nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Dropout(0.5),  
        nn.Linear(4096, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU()
    ),

    
    nn.Sequential(
        nn.Linear(4096, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.4), 
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        
    ),

   
    nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        
    ),

    
    nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),  
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        
    ),

    
    nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.1),  
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        
    ),

   
    nn.Linear(256, 1)
).to(device)

# -- Load emotion Model
emotion_weights = torchvision.models.VGG19_Weights.DEFAULT
emotion_model = torchvision.models.vgg19( weights=emotion_weights)

# Freeze all parameters in the feature layers
for param in emotion_model.features.parameters():
    param.requires_grad = False
    
# Unfreeze only the last 20 layers of the feature layers
for param in emotion_model.features[-20:].parameters():
    param.requires_grad = True    


# Modify the classifier
emotion_model.classifier = nn.Sequential(
    
    nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(4096, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU()
    ),

    
    nn.Sequential(
        nn.Linear(4096, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        
    ),

    
    nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        
    ),

    
    nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        
    ),

    
    nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
       
    ),

    
    nn.Linear(256, 7)  # 7 emotion classes
).to(device)




# Resolve model paths relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

gender_model.load_state_dict(torch.load(os.path.join(_SCRIPT_DIR, "fine_tuned_gender_model.pth"), map_location=device))
gender_model.to(device).eval()

age_model.load_state_dict(torch.load(os.path.join(_SCRIPT_DIR, "fine_tuned_age_model.pth"), map_location=device))
age_model.to(device).eval()

emotion_model.load_state_dict(torch.load(os.path.join(_SCRIPT_DIR, "emotion_model_fine_tuned.pth"), map_location=device))
emotion_model.to(device).eval()

# Data transformation for gender model
gender_infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data transformation for age model
age_infer_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data transformation for emotion model
emotion_infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Helper function to predict and display results
def predict_and_display(frame, face_box, gender_model, age_model, emotion_model, all_face_boxes=None):
    x, y, w, h = face_box
    face = frame[y:y+h, x:x+w]
    face_pil = Image.fromarray(face)

    # Process the face image for each model
    gender_tensor = gender_infer_transform(face_pil).unsqueeze(0).to(device)
    age_tensor = age_infer_transform(face_pil).unsqueeze(0).to(device)
    emotion_tensor = emotion_infer_transform(face_pil).unsqueeze(0).to(device)

    # Predictions
    with torch.no_grad():
        gender_output = gender_model(gender_tensor)
        age_output = age_model(age_tensor)
        emotion_output = emotion_model(emotion_tensor)

        # Postprocess predictions
        gender_pred = torch.round(torch.sigmoid(gender_output))
        gender_index = int(gender_pred.item())
        gender_labels = ['Male', 'Female']
        gender_text = gender_labels[gender_index]

        age_text = int(age_output.item())

        emotion_index = torch.argmax(emotion_output, dim=1).item()
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_text = emotion_labels[emotion_index]

    # Annotate the frame with the predictions
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display results vertically above or below the bounding box
    labels = [
        f"Gender: {gender_text}",
        f"Age: {age_text}",
        f"Emotion: {emotion_text}"
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    line_spacing = 8
    padding = 10

    # Calculate dimensions of the text block
    line_heights = []
    line_widths = []
    for line in labels:
        (lw, lh), _ = cv2.getTextSize(line, font, font_scale, thickness)
        line_heights.append(lh)
        line_widths.append(lw)
    
    total_text_height = sum(line_heights) + line_spacing * (len(labels) - 1)
    space_needed = total_text_height + padding

    # Decide position: 'top' or 'bottom'
    top_fits = (y - space_needed >= 0)
    bottom_fits = (y + h + space_needed <= frame.shape[0])
    
    top_overlaps = False
    bottom_overlaps = False
    
    if all_face_boxes and len(all_face_boxes) > 1:
        top_ymin, top_ymax = y - space_needed, y
        top_xmin, top_xmax = x - 5, x + w + 5
        
        bottom_ymin, bottom_ymax = y + h, y + h + space_needed
        bottom_xmin, bottom_xmax = x - 5, x + w + 5
        
        for ob in all_face_boxes:
            if ob == face_box:
                continue
            ox, oy, ow, oh = ob
            # Check overlap with top area
            if (ox < top_xmax and ox + ow > top_xmin and 
                oy < top_ymax and oy + oh > top_ymin):
                top_overlaps = True
            # Check overlap with bottom area
            if (ox < bottom_xmax and ox + ow > bottom_xmin and 
                oy < bottom_ymax and oy + oh > bottom_ymin):
                bottom_overlaps = True

    # Decide placement based on space and overlap
    if top_fits and not top_overlaps:
        position = 'top'
    elif bottom_fits and not bottom_overlaps:
        position = 'bottom'
    elif top_fits and bottom_fits:
        position = 'top'
    elif top_fits:
        position = 'top'
    elif bottom_fits:
        position = 'bottom'
    else:
        position = 'top'

    # Draw the text block
    if position == 'top':
        current_y = y - padding - total_text_height
    else:
        current_y = y + h + padding

    for i, line in enumerate(labels):
        lh = line_heights[i]
        lw = line_widths[i]
        
        # Left-align with face box, but keep within frame boundaries
        text_x = x
        if text_x + lw > frame.shape[1] - 10:
            text_x = max(10, frame.shape[1] - lw - 10)
            
        cv2.putText(frame, line, (text_x, current_y + lh), 
                    font, font_scale, (255, 0, 0), thickness)
        current_y += lh + line_spacing

    return frame

# Streamlit app structure
st.title("Webcam Face Prediction")
st.sidebar.write("This app uses your webcam to detect gender, age, and emotion.")

camera_input = st.camera_input("Capture Image")

if camera_input:
    # Convert the camera input (PIL image) to OpenCV format (numpy array)
    img = Image.open(camera_input)
    img = np.array(img)

    # Initialize face detection model from Mediapipe
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
        # Image from st.camera_input is already RGB
        results = face_detection.process(img)

        # Process face detections
        if results.detections:
            face_boxes = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * img.shape[1])
                y = int(bbox.ymin * img.shape[0])
                w = int(bbox.width * img.shape[1])
                h = int(bbox.height * img.shape[0])

                # Clamp values to image size
                x, y = max(0, x), max(0, y)
                w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
                face_boxes.append((x, y, w, h))

            for face_box in face_boxes:
                # Make predictions and display
                img = predict_and_display(img, face_box, gender_model, age_model, emotion_model, face_boxes)

    # Display the image with the annotations
    st.image(img, caption="Predicted Image", channels="RGB")
