import os
import cv2
import pytesseract
from ultralytics import YOLO
from gtts import gTTS
import shutil

# Helper function for image processing
# Note: Passing the model instance from main.py to avoid reloading
def process_newspaper_image(image_path: str, language: str, output_base_dir: str, model: YOLO):
    if model is None:
        raise RuntimeError("YOLO model not loaded.")

    results = model(image_path)
    original_image = cv2.imread(image_path)

    processed_items = []

    # Check if results[0] contains boxes
    if results and results[0].boxes:
        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            confidence = results[0].boxes.conf[i]
            class_id = int(results[0].boxes.cls[i])
            class_name = model.names[class_id]

            if confidence > 0.1 and class_name == 'news':
                # Crop image
                cropped_img = original_image[y1:y2, x1:x2]
                
                # Ensure output directory exists before saving
                os.makedirs(output_base_dir, exist_ok=True)
                
                cropped_img_filename = f"cropped_box_{i}.jpg"
                cropped_img_path_abs = os.path.join(output_base_dir, cropped_img_filename)
                cv2.imwrite(cropped_img_path_abs, cropped_img)

                # OCR
                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                text = pytesseract.image_to_string(
                    cropped_img_rgb,
                    lang='eng' if language.lower() == 'english' else 'hin' # Use 'eng' or 'hin' for tesseract
                )
                cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

                # Generate audio
                if cleaned_text.strip():
                    audio_filename = f"output_box_{i}.mp3"
                    audio_file_path_abs = os.path.join(output_base_dir, audio_filename)
                    
                    # Determine gTTS language code
                    gtts_lang = 'en' if language.lower() == 'english' else 'hi'
                    
                    tts = gTTS(
                        text=cleaned_text,
                        lang=gtts_lang,
                        slow=False # Optional: adjust speed
                    )
                    tts.save(audio_file_path_abs)
                    
                    audio_path_url = f"/processed_files/{os.path.basename(output_base_dir)}/{audio_filename}"
                else:
                    audio_path_url = None # No audio if no text

                processed_items.append(
                    {
                        "text": cleaned_text,
                        "audio_path": audio_path_url,
                        "image_path": f"/processed_files/{os.path.basename(output_base_dir)}/{cropped_img_filename}"
                    }
                )

    return processed_items