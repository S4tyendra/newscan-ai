import os
import cv2
import pytesseract
from ultralytics import YOLO
from gtts import gTTS
from langdetect import detect, DetectorFactory # Import detect and DetectorFactory
DetectorFactory.seed = 0 # Set seed for reproducibility
import shutil
import asyncio
import logging

# Setup logging for the helper module
logger = logging.getLogger(__name__)

# Helper function for processing a single detected box
async def process_single_box(
    box_index: int,
    box,
    original_image,
    output_base_dir: str,
    model_names: dict,
    upload_id_str: str,
    task_status: dict,
    language: str # Add language parameter
):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = box.conf[0]
    class_id = int(box.cls[0])
    class_name = model_names[class_id]

    box_log_prefix = f"Task {upload_id_str} Box {box_index}:"
    logger.info(f"{box_log_prefix} Processing box.")
    task_status[upload_id_str]["logs"].append(f"{box_log_prefix} Processing box...")

    if confidence > 0.1 and class_name == 'news':
        try:
            # Crop image
            cropped_img = original_image[y1:y2, x1:x2]

            # Ensure output directory exists before saving
            os.makedirs(output_base_dir, exist_ok=True)

            cropped_img_filename = f"cropped_box_{box_index}.jpg"
            cropped_img_path_abs = os.path.join(output_base_dir, cropped_img_filename)
            cv2.imwrite(cropped_img_path_abs, cropped_img)
            logger.info(f"{box_log_prefix} Cropped image saved.")
            task_status[upload_id_str]["logs"].append(f"{box_log_prefix} Cropped image saved.")

            # OCR - Use the provided language, mapping to Tesseract code
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

            # Map input language code to Tesseract language code
            tesseract_lang_map = {
                'en': 'eng',
                'hi': 'hin',
                'te': 'tel'
            }
            tesseract_lang = tesseract_lang_map.get(language, 'eng') # Default to English if code is unknown

            # Use the mapped Tesseract language for OCR
            cleaned_text = pytesseract.image_to_string(cropped_img_rgb, lang=tesseract_lang)
            cleaned_text = "\n".join(line.strip() for line in cleaned_text.splitlines() if line.strip())

            # Store the original selected language
            detected_language = language
            print(f"{box_log_prefix} Detected language: {detected_language}")
            task_status[upload_id_str]["logs"].append(f"{box_log_prefix} Detected language: {detected_language}")

            logger.info(f"{box_log_prefix} OCR completed. Text length: {len(cleaned_text)}. Detected language: {detected_language}")
            task_status[upload_id_str]["logs"].append(f"{box_log_prefix} Text extracted. Language: {detected_language}")

            # Do NOT generate audio here initially. Audio generation will be on-demand.
            audio_path_url = None

            # Store results for this box
            return {
                "box_index": box_index, # Add box index for on-demand actions
                "text": cleaned_text,
                "language": detected_language, # Store the detected language
                "audio_path": audio_path_url, # Initially None
                "image_path": f"/processed_files/{os.path.basename(output_base_dir)}/{cropped_img_filename}",
                "summary": None # Initially None for on-demand summary
            }

        except Exception as e:
            logger.error(f"{box_log_prefix} Error processing box: {e}")
            task_status[upload_id_str]["logs"].append(f"{box_log_prefix} Error processing box: {e}")
            return {
                "box_index": box_index,
                "text": f"Error processing box: {e}",
                "language": 'unknown', # Indicate unknown language on error
                "audio_path": None,
                "image_path": None, # Or a placeholder for error image
                "summary": None,
                "error": str(e)
            }
    else:
        logger.info(f"{box_log_prefix} Skipping box due to low confidence or wrong class.")
        task_status[upload_id_str]["logs"].append(f"{box_log_prefix} Skipping box (confidence or class).")
        return None # Skip if not a news box or low confidence


# Main image processing function
async def process_newspaper_image(
    image_path: str,
    output_base_dir: str,
    model: YOLO,
    upload_id_str: str,
    task_status: dict,
    language: str # Add language parameter
):
    if model is None:
        raise RuntimeError("YOLO model not loaded.")

    # Run detection asynchronously (YOLO .predict supports async)
    # Note: Depending on the YOLO version and backend, the prediction itself might still be blocking.
    # For true async, consider running detection in a thread pool if it's a bottleneck.
    # For simplicity, let's assume the predict call is reasonably fast or can be awaited.
    logger.info(f"Task {upload_id_str}: Starting YOLO detection.")
    task_status[upload_id_str]["logs"].append("Starting YOLO detection...")
    results = await asyncio.to_thread(model.predict, image_path) # Run the blocking predict call in a thread pool

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not read image file: {image_path}")

    processed_items = []
    box_tasks = []

    # Check if results[0] contains boxes
    if results and results[0].boxes:
        logger.info(f"Task {upload_id_str}: Detected {len(results[0].boxes)} boxes.")
        task_status[upload_id_str]["logs"].append(f"Detected {len(results[0].boxes)} potential news sections.")
        total_boxes = len(results[0].boxes)

        for i in range(total_boxes):
            box = results[0].boxes[i] # Access box using index
            # Create a task for each box
            box_tasks.append(
                asyncio.create_task(
                    process_single_box(
                        i,
                        box,
                        original_image,
                        output_base_dir,
                        model.names,
                        upload_id_str,
                        task_status,
                        language # Pass the language to process_single_box
                    )
                )
            )

        # Wait for all box tasks to complete and collect results
        completed_tasks = 0
        for task in asyncio.as_completed(box_tasks):
            result = await task
            if result: # Only append if result is not None (box was processed)
                processed_items.append(result)

            completed_tasks += 1
            # Update progress
            task_status[upload_id_str]["progress"] = completed_tasks / total_boxes
            task_status[upload_id_str]["logs"].append(f"Processed {completed_tasks}/{total_boxes} boxes. Progress: {(task_status[upload_id_str]['progress'] * 100):.0f}%")
            logger.info(f"Task {upload_id_str}: Processed {completed_tasks}/{total_boxes} boxes.")

    else:
        logger.info(f"Task {upload_id_str}: No boxes detected.")
        task_status[upload_id_str]["logs"].append("No news sections detected in the image.")

    # Sort processed_items by box_index before returning
    processed_items.sort(key=lambda x: x.get("box_index", float('inf')))

    logger.info(f"Task {upload_id_str}: Finished processing all boxes.")
    task_status[upload_id_str]["logs"].append("Finished processing.")

    return processed_items