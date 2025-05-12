from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import pytesseract
from gtts import gTTS
from ultralytics import YOLO
import os
from typing import List
import shutil
from pydantic import BaseModel

app = FastAPI(
    title="Newspaper Text Detection and Recognition",
    description="API for detecting and recognizing text in newspaper images",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_PATH = "best_30_epochs.pt"
OUTPUT_DIR = "cropped_boxes"
UPLOAD_DIR = "uploads"

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize YOLO model
model = YOLO(MODEL_PATH)

# Mount static files
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")

class ProcessResponse(BaseModel):
    text: str
    audio_path: str
    image_path: str

@app.post("/process/", response_model=List[ProcessResponse])
async def process_image(
    file: UploadFile = File(...),
    language: str = Form(...),
):
    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process image with YOLO
        results = model(file_path)
        original_image = cv2.imread(file_path)

        processed_results = []

        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            confidence = results[0].boxes.conf[i]
            class_id = int(results[0].boxes.cls[i])
            class_name = model.names[class_id]

            if confidence > 0.1 and class_name == 'news':
                # Crop image
                cropped_img = original_image[y1:y2, x1:x2]
                cropped_img_path = os.path.join(OUTPUT_DIR, f"cropped_box_{i}.jpg")
                cv2.imwrite(cropped_img_path, cropped_img)

                # OCR
                cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                text = pytesseract.image_to_string(
                    cropped_img_rgb, 
                    lang=language
                )
                cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

                # Generate audio
                audio_file = os.path.join(OUTPUT_DIR, f"output_box_{i}.mp3")
                tts = gTTS(
                    text=cleaned_text,
                    lang='en' if language.lower() == 'english' else 'hi'
                )
                tts.save(audio_file)

                processed_results.append(
                    ProcessResponse(
                        text=cleaned_text,
                        audio_path=f"/static/output_box_{i}.mp3",
                        image_path=f"/static/cropped_box_{i}.jpg"
                    )
                )

        return processed_results

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/")
async def root():
    return {"message": "Welcome to Newspaper Text Detection and Recognition API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8008, reload=True)
