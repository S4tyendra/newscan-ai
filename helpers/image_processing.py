import os
import cv2
import pytesseract
from ultralytics import YOLO
from gtts import gTTS
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

import shutil
import asyncio
import logging


logger = logging.getLogger(__name__)


async def process_single_box(
    box_index: int,
    box,
    original_image,
    output_base_dir: str,
    model_names: dict,
    upload_id_str: str,
    task_status: dict,
    language: str,
):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = box.conf[0]
    class_id = int(box.cls[0])
    class_name = model_names[class_id]

    box_log_prefix = f"Task {upload_id_str} Box {box_index}:"
    logger.info(f"{box_log_prefix} Processing box.")
    task_status[upload_id_str]["logs"].append(f"{box_log_prefix} Processing box...")

    if confidence > 0.1 and class_name == "news":
        try:

            cropped_img = original_image[y1:y2, x1:x2]

            os.makedirs(output_base_dir, exist_ok=True)

            cropped_img_filename = f"cropped_box_{box_index}.jpg"
            cropped_img_path_abs = os.path.join(output_base_dir, cropped_img_filename)
            cv2.imwrite(cropped_img_path_abs, cropped_img)
            logger.info(f"{box_log_prefix} Cropped image saved.")
            task_status[upload_id_str]["logs"].append(
                f"{box_log_prefix} Cropped image saved."
            )

            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

            tesseract_lang_map = {"en": "eng", "hi": "hin", "te": "tel"}
            tesseract_lang = tesseract_lang_map.get(language, "eng")

            cleaned_text = pytesseract.image_to_string(
                cropped_img_rgb, lang=tesseract_lang
            )
            cleaned_text = "\n".join(
                line.strip() for line in cleaned_text.splitlines() if line.strip()
            )

            detected_language = language
            print(f"{box_log_prefix} Detected language: {detected_language}")
            task_status[upload_id_str]["logs"].append(
                f"{box_log_prefix} Detected language: {detected_language}"
            )

            logger.info(
                f"{box_log_prefix} OCR completed. Text length: {len(cleaned_text)}. Detected language: {detected_language}"
            )
            task_status[upload_id_str]["logs"].append(
                f"{box_log_prefix} Text extracted. Language: {detected_language}"
            )

            audio_path_url = None

            return {
                "box_index": box_index,
                "text": cleaned_text,
                "language": detected_language,
                "audio_path": audio_path_url,
                "image_path": f"/processed_files/{os.path.basename(output_base_dir)}/{cropped_img_filename}",
                "summary": None,
            }

        except Exception as e:
            logger.error(f"{box_log_prefix} Error processing box: {e}")
            task_status[upload_id_str]["logs"].append(
                f"{box_log_prefix} Error processing box: {e}"
            )
            return {
                "box_index": box_index,
                "text": f"Error processing box: {e}",
                "language": "unknown",
                "audio_path": None,
                "image_path": None,
                "summary": None,
                "error": str(e),
            }
    else:
        logger.info(
            f"{box_log_prefix} Skipping box due to low confidence or wrong class."
        )
        task_status[upload_id_str]["logs"].append(
            f"{box_log_prefix} Skipping box (confidence or class)."
        )
        return None


async def process_newspaper_image(
    image_path: str,
    output_base_dir: str,
    model: YOLO,
    upload_id_str: str,
    task_status: dict,
    language: str,
):
    if model is None:
        raise RuntimeError("YOLO model not loaded.")

    logger.info(f"Task {upload_id_str}: Starting YOLO detection.")
    task_status[upload_id_str]["logs"].append("Starting YOLO detection...")
    results = await asyncio.to_thread(model.predict, image_path)

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not read image file: {image_path}")

    processed_items = []
    box_tasks = []

    if results and results[0].boxes:
        logger.info(f"Task {upload_id_str}: Detected {len(results[0].boxes)} boxes.")
        task_status[upload_id_str]["logs"].append(
            f"Detected {len(results[0].boxes)} potential news sections."
        )
        total_boxes = len(results[0].boxes)

        for i in range(total_boxes):
            box = results[0].boxes[i]

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
                        language,
                    )
                )
            )

        completed_tasks = 0
        for task in asyncio.as_completed(box_tasks):
            result = await task
            if result:

                processed_items.append(result)

            completed_tasks += 1

            task_status[upload_id_str]["progress"] = completed_tasks / total_boxes
            task_status[upload_id_str]["logs"].append(
                f"Processed {completed_tasks}/{total_boxes} boxes. Progress: {(task_status[upload_id_str]['progress'] * 100):.0f}%"
            )
            logger.info(
                f"Task {upload_id_str}: Processed {completed_tasks}/{total_boxes} boxes."
            )

    else:
        logger.info(f"Task {upload_id_str}: No boxes detected.")
        task_status[upload_id_str]["logs"].append(
            "No news sections detected in the image."
        )

    processed_items.sort(key=lambda x: x.get("box_index", float("inf")))

    logger.info(f"Task {upload_id_str}: Finished processing all boxes.")
    task_status[upload_id_str]["logs"].append("Finished processing.")

    return processed_items
