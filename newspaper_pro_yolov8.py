import cv2
import os
import pytesseract
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Paths
model_path = "best_30_epochs.pt"  # Update this to your local model path
input_image_path = "IMG_20240904_171353626.jpg"  # Update this to your image path
output_dir = "cropped_boxes"

# Load the YOLO model
model = YOLO(model_path)

# Create a directory to save cropped images
os.makedirs(output_dir, exist_ok=True)

# Perform inference on an image
results = model(input_image_path)

# Get the original image for cropping
original_image = cv2.imread(input_image_path)

# Loop over the detected objects
total_boxes = len(results[0].boxes.xyxy)
print(f"Total boxes detected: {total_boxes}")

for i, box in enumerate(results[0].boxes.xyxy):
    print(f"Processing box {i}...")  # Debugging statement

    x1, y1, x2, y2 = map(int, box)
    confidence = results[0].boxes.conf[i]  # Get confidence score
    class_id = results[0].boxes.cls[i]  # Get class ID

    if confidence > 0.1:  # Threshold for confidence
        # Crop the image using the bounding box coordinates
        cropped_img = original_image[y1:y2, x1:x2]

        # Save the cropped image
        output_path = os.path.join(output_dir, f"box_{i}.jpg")
        cv2.imwrite(output_path, cropped_img)

        # Perform OCR on the cropped image
        text = pytesseract.image_to_string(cropped_img)

        # Clean up the extracted text
        cleaned_text = "\n".join(line.strip().replace(":", "") for line in text.splitlines() if line.strip())

        # Display the cropped image using OpenCV
        cv2.imshow(f'Box {i}', cropped_img)
        cv2.waitKey(1000)  # Wait for 1 second (1000 milliseconds) before closing the window

        # Print the cleaned extracted text
        print(f"Text from box {i}:")
        print(cleaned_text)
        print()  # Print a newline for better readability

cv2.destroyAllWindows()  # Close all OpenCV windows
