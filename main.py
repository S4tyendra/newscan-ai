import os
import shutil
import uuid
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uuid
from ultralytics import YOLO
from helpers.image_processing import process_newspaper_image
import logging
import json
from gtts import gTTS
import google.generativeai as genai
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Newspaper Text Detection and Recognition",
    description="Web application for detecting and recognizing text in newspaper images",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount(
    "/processed_files", StaticFiles(directory="processed_files"), name="processed_files"
)
MODEL_PATH = "best_30_epochs.pt"
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed_files"
STATUS_FILE = "status.json"
LOG_FILE = "logs.json"

GEMINI_API_KEY = "AIzaSyBVVKW7jJaxndA04RDgqR0WxmyRA5ajby4"
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"

genai.configure(api_key=GEMINI_API_KEY)

try:
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    logger.info(f"Gemini model '{GEMINI_MODEL_NAME}' loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Gemini model '{GEMINI_MODEL_NAME}': {e}")
    gemini_model = None
LOG_FILE = "logs.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

model = None
try:
    model = YOLO(MODEL_PATH)
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")

task_queue = asyncio.Queue()

task_status = {}

async def process_task_worker():
    while True:
        task_data = await task_queue.get()
        upload_id = task_data["upload_id"]
        file_path_temp = task_data["file_path_temp"]
        processed_output_dir = task_data["processed_output_dir"]
        upload_id_str = str(upload_id)
        task_status[upload_id_str]["status"] = "processing"
        task_status[upload_id_str]["logs"].append("Task started.")
        logger.info(f"Processing task {upload_id}")
        try:
            if model is None:
                task_status[upload_id_str]["status"] = "failed"
                task_status[upload_id_str][
                    "error"
                ] = "Image processing model failed to load on startup."
                task_status[upload_id_str]["logs"].append(
                    "Task failed: Image processing model not loaded."
                )
                logger.error(f"Task {upload_id_str} failed: YOLO model not loaded.")
            else:
                processed_results = await process_newspaper_image(
                    file_path_temp,
                    processed_output_dir,
                    model,
                    upload_id_str,
                    task_status,
                    task_data.get("language", "en"),
                )
                task_status[upload_id_str]["status"] = "completed"
                task_status[upload_id_str]["results"] = processed_results
                task_status[upload_id_str]["logs"].append(
                    "Task completed successfully."
                )
                logger.info(f"Task {upload_id} completed.")
        except Exception as e:
            task_status[upload_id_str]["status"] = "failed"
            task_status[upload_id_str]["error"] = str(e)
            task_status[upload_id_str]["logs"].append(f"Task failed: {e}")
            logger.error(f"Task {upload_id} failed: {e}")
        finally:
            if os.path.exists(os.path.dirname(file_path_temp)):
                shutil.rmtree(os.path.dirname(file_path_temp))
                task_status[upload_id_str]["logs"].append(
                    "Cleaned up temporary upload directory."
                )
            task_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_task_worker())
    logger.info("Background task worker started.")

@app.get("/", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/")
async def process_upload(
    request: Request, file: UploadFile = File(...), language: str = Form("en")
):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={
                "upload_id": "error",
                "error_message": "Image processing model failed to load. Please check server logs.",
            },
        )
    upload_id = uuid.uuid4()
    upload_id_str = str(upload_id)
    upload_dir_temp = os.path.join(UPLOAD_DIR, upload_id_str)
    processed_output_dir = os.path.join(PROCESSED_DIR, upload_id_str)
    os.makedirs(upload_dir_temp, exist_ok=True)
    os.makedirs(processed_output_dir, exist_ok=True)
    if file.filename is None:
        if os.path.exists(upload_dir_temp):
            shutil.rmtree(upload_dir_temp)
        if os.path.exists(processed_output_dir):
            shutil.rmtree(processed_output_dir)
        return JSONResponse(
            status_code=400,
            content={
                "upload_id": "error",
                "error_message": "No filename provided for the uploaded file.",
            },
        )
    file_path_temp = os.path.join(upload_dir_temp, file.filename)
    try:
        with open(file_path_temp, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        task_status[upload_id_str] = {
            "status": "pending",
            "progress": 0.0,
            "logs": [f"Received file: {file.filename}. Adding to queue."],
            "results": [],
            "error": None,
            "language": language,
        }
        logger.info(
            f"Received file {file.filename}, assigning ID {upload_id_str}, adding to queue with language '{language}'."
        )
        await task_queue.put(
            {
                "upload_id": upload_id,
                "file_path_temp": file_path_temp,
                "processed_output_dir": processed_output_dir,
                "language": language,
            }
        )
        return RedirectResponse(url=f"/status/{upload_id_str}", status_code=303)
    except Exception as e:
        if os.path.exists(upload_dir_temp):
            shutil.rmtree(upload_dir_temp)
        if os.path.exists(processed_output_dir):
            shutil.rmtree(processed_output_dir)
        error_id = str(uuid.uuid4())
        task_status[error_id] = {
            "status": "failed",
            "progress": 0.0,
            "logs": [f"Failed to receive or queue file: {e}"],
            "results": [],
            "error": str(e),
        }
        logger.error(f"Failed to receive or queue file: {e}")
        return JSONResponse(
            status_code=500, content={"upload_id": error_id, "error_message": str(e)}
        )

@app.get("/status/{upload_id}", response_class=HTMLResponse)
async def get_status_page(request: Request, upload_id: str):
    if upload_id not in task_status:
        if not os.path.exists(os.path.join(PROCESSED_DIR, upload_id)):
            return templates.TemplateResponse(
                "error.html",
                {
                    "request": request,
                    "error_message": f"Invalid or expired Upload ID: {upload_id}",
                },
                status_code=404,
            )
    return templates.TemplateResponse(
        "status.html", {"request": request, "upload_id": upload_id}
    )

@app.get("/api/status/{upload_id}")
async def get_task_status(upload_id: str):
    status = task_status.get(upload_id)
    if status:
        return JSONResponse(status)
    if os.path.exists(os.path.join(PROCESSED_DIR, upload_id)):
        return JSONResponse(
            status_code=404,
            content={
                "status": "unknown",
                "error": "Task status not found, but directory exists. Server may have restarted.",
            },
        )
    return JSONResponse(status_code=404, content={"status": "not_found"})

@app.get("/api/logs/{upload_id}")
async def get_task_logs(upload_id: str):
    status = task_status.get(upload_id)
    if status:
        return JSONResponse({"logs": status.get("logs", [])})
    return JSONResponse(status_code=404, content={"logs": []})

@app.get("/api/generate_audio/{upload_id}/{box_index}")
async def generate_audio(upload_id: str, box_index: int):
    task = task_status.get(upload_id)
    if not task or task["status"] != "completed":
        return JSONResponse(
            status_code=404, content={"error": "Task not found or not completed."}
        )
    box_result = None
    for item in task["results"]:
        if item.get("box_index") == box_index:
            box_result = item
            break
    if not box_result or not box_result.get("text"):
        return JSONResponse(
            status_code=404, content={"error": "Box not found or no text extracted."}
        )
    cleaned_text = box_result["text"]
    selected_language = task.get("language", "en")
    gtts_lang = "en"
    if selected_language == "hi":
        gtts_lang = "hi"
    elif selected_language == "te":
        gtts_lang = "te"
    processed_output_dir = os.path.join(PROCESSED_DIR, upload_id)
    audio_filename = f"output_box_{box_index}.mp3"
    audio_file_path_abs = os.path.join(processed_output_dir, audio_filename)
    if os.path.exists(audio_file_path_abs):
        audio_path_url = f"/processed_files/{upload_id}/{audio_filename}"
        logger.info(
            f"Audio already exists for task {upload_id}, box {box_index}. Returning existing."
        )
        return JSONResponse({"audio_path": audio_path_url})
    try:
        tts = gTTS(text=cleaned_text, lang=gtts_lang, slow=False)
        tts.save(audio_file_path_abs)
        audio_path_url = f"/processed_files/{upload_id}/{audio_filename}"
        logger.info(f"Generated audio for task {upload_id}, box {box_index}.")
        box_result["audio_path"] = audio_path_url
        return JSONResponse({"audio_path": audio_path_url})
    except Exception as e:
        logger.error(
            f"Error generating audio for task {upload_id}, box {box_index}: {e}"
        )
        return JSONResponse(
            status_code=500, content={"error": f"Failed to generate audio: {e}"}
        )

@app.get("/api/summarize/{upload_id}/{box_index}")
async def summarize_text(upload_id: str, box_index: int):
    if gemini_model is None:
        return JSONResponse(
            status_code=500, content={"error": "Gemini model failed to load."}
        )
    task = task_status.get(upload_id)
    if not task or task["status"] != "completed":
        return JSONResponse(
            status_code=404, content={"error": "Task not found or not completed."}
        )
    box_result = None
    for item in task["results"]:
        if item.get("box_index") == box_index:
            box_result = item
            break
    if not box_result or not box_result.get("text"):
        return JSONResponse(
            status_code=404, content={"error": "Box not found or no text extracted."}
        )
    cleaned_text = box_result["text"]
    if box_result.get("summary"):
        logger.info(
            f"Summary already exists for task {upload_id}, box {box_index}. Returning existing."
        )
        return JSONResponse({"summary": box_result["summary"]})
    try:
        prompt = f"Summarize the following text:\n\n{cleaned_text}"
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        summary = response.text.strip()
        logger.info(f"Generated summary for task {upload_id}, box {box_index}.")
        box_result["summary"] = summary
        return JSONResponse({"summary": summary})
    except Exception as e:
        logger.error(
            f"Error generating summary for task {upload_id}, box {box_index}: {e}"
        )
        return JSONResponse(
            status_code=500, content={"error": f"Failed to generate summary: {e}"}
        )

@app.get("/list_uploads/", response_class=HTMLResponse)
async def list_uploads(request: Request):
    upload_ids = [
        d
        for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    ]
    return templates.TemplateResponse(
        "list_uploads.html", {"request": request, "upload_ids": upload_ids}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8008)
