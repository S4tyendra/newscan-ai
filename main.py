import os
import shutil
import uuid
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware # Needed for frontend polling if on different port/host
import os
import shutil
import uuid
from ultralytics import YOLO # Assuming YOLO is needed for model loading in main
from helpers.image_processing import process_newspaper_image # Import the helper function
import logging
import json # To store results and logs
from gtts import gTTS # Import gTTS for audio generation
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Newspaper Text Detection and Recognition",
    description="Web application for detecting and recognizing text in newspaper images",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to poll from a different origin during development
# In production, configure this more restrictively
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)


# Configure Templates
templates = Jinja2Templates(directory="templates")

# Mount static files and processed files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/processed_files", StaticFiles(directory="processed_files"), name="processed_files")
# Constants
MODEL_PATH = "best_30_epochs.pt" # Ensure this file is in the correct location
UPLOAD_DIR = "uploads" # Temporary storage for uploads
PROCESSED_DIR = "processed_files" # Stores processed results in UUID dirs
STATUS_FILE = "status.json" # File to store task status and results
LOG_FILE = "logs.json" # File to store logs per task

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyBVVKW7jJaxndA04RDgqR0WxmyRA5ajby4" # Replace with your actual API key, consider using environment variables
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite" # Use the specified model

# Configure the gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model (can be done once or per request, depending on usage patterns and limits)
# Doing it once here for simplicity
try:
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    logger.info(f"Gemini model '{GEMINI_MODEL_NAME}' loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Gemini model '{GEMINI_MODEL_NAME}': {e}")
    gemini_model = None # Set to None if loading fails

LOG_FILE = "logs.json" # File to store logs per task

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Initialize YOLO model (Load once at startup)
model = None
try:
    model = YOLO(MODEL_PATH)
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    # Depending on criticality, you might want to raise the exception or handle it differently
    # model remains None if loading fails

# --- Async Processing Setup ---
# Queue for tasks
task_queue = asyncio.Queue()

# Dictionary to store task status, results, and logs
# Structure:
# {
#   uuid_str: {
#     "status": "pending" | "processing" | "completed" | "failed",
#     "progress": 0.0, # Optional: percentage
#     "logs": ["log message 1", "log message 2"],
#     "results": [...], # Processed items list on completion
#     "error": "error message" # On failure
#   }
# }
task_status = {}

# Background worker to process tasks
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
                 task_status[upload_id_str]["error"] = "Image processing model failed to load on startup."
                 task_status[upload_id_str]["logs"].append("Task failed: Image processing model not loaded.")
                 logger.error(f"Task {upload_id_str} failed: YOLO model not loaded.")
                 # No need to call process_newspaper_image, just exit this try block
            else:
                # Process the image using the helper function, passing the model instance
                # The helper function needs to be modified to handle parallelism and logging
                # Process the image using the helper function, passing the model instance and language
                processed_results = await process_newspaper_image(
                    file_path_temp,
                    processed_output_dir,
                    model,
                    upload_id_str, # Pass upload_id for logging/status updates from helper
                    task_status, # Pass task_status dictionary to helper for updates
                    task_data.get("language", "en") # Pass the selected language, default to 'en'
                )

                task_status[upload_id_str]["status"] = "completed"
                task_status[upload_id_str]["results"] = processed_results
                task_status[upload_id_str]["logs"].append("Task completed successfully.")
                logger.info(f"Task {upload_id} completed.")

        except Exception as e:
            task_status[upload_id_str]["status"] = "failed"
            task_status[upload_id_str]["error"] = str(e)
            task_status[upload_id_str]["logs"].append(f"Task failed: {e}")
            logger.error(f"Task {upload_id} failed: {e}")

        finally:
            # Clean up the temporary uploaded file and its directory
            if os.path.exists(os.path.dirname(file_path_temp)):
                 shutil.rmtree(os.path.dirname(file_path_temp))
                 task_status[upload_id_str]["logs"].append("Cleaned up temporary upload directory.")

            # Save task status and results
            # You might want to save task_status periodically or after completion/failure
            # For now, let's save on task completion/failure
            # save_task_status() # Implement a save function if needed for persistence

            task_queue.task_done()

# Start the worker when the application starts
@app.on_event("startup")
async def startup_event():
    # Optionally load previous task status here
    # load_task_status()
    asyncio.create_task(process_task_worker())
    logger.info("Background task worker started.")

# --- API Endpoints ---

# Root endpoint - serves the upload form
@app.get("/", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Process endpoint - handles image upload and adds to queue
@app.post("/process/")
async def process_upload(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("en") # Add language form data with a default
):
    if model is None:
         # Note: Client will navigate to /status/error_uuid if model fails,
         # so this direct template response won't be used via the frontend redirect logic.
         # Need a way to handle this in the status endpoint or redirect there with an error flag.
         # For now, returning an error JSON.
         return JSONResponse(
             status_code=500,
             content={"upload_id": "error", "error_message": "Image processing model failed to load. Please check server logs."}
         )

    upload_id = uuid.uuid4()
    upload_id_str = str(upload_id)
    upload_dir_temp = os.path.join(UPLOAD_DIR, upload_id_str) # Use a temp dir for original upload
    processed_output_dir = os.path.join(PROCESSED_DIR, upload_id_str) # Dir for results

    os.makedirs(upload_dir_temp, exist_ok=True)
    os.makedirs(processed_output_dir, exist_ok=True)

    if file.filename is None:
        # Clean up created directories on error
        if os.path.exists(upload_dir_temp): shutil.rmtree(upload_dir_temp)
        if os.path.exists(processed_output_dir): shutil.rmtree(processed_output_dir)
        return JSONResponse(
            status_code=400,
            content={"upload_id": "error", "error_message": "No filename provided for the uploaded file."}
        )

    file_path_temp = os.path.join(upload_dir_temp, file.filename)

    try:
        # Save the uploaded file temporarily
        with open(file_path_temp, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Initialize status for this task
        task_status[upload_id_str] = {
            "status": "pending",
            "progress": 0.0,
            "logs": [f"Received file: {file.filename}. Adding to queue."],
            "results": [],
            "error": None,
            "language": language # Store the selected language in task status
        }
        logger.info(f"Received file {file.filename}, assigning ID {upload_id_str}, adding to queue with language '{language}'.")
        # Add task to the queue
        await task_queue.put({
            "upload_id": upload_id,
            "file_path_temp": file_path_temp,
            "processed_output_dir": processed_output_dir,
            "language": language # Pass language to task data
        })

        # Return the upload ID immediately
        # The frontend will use this ID to navigate to the status page
        return RedirectResponse(
            url=f"/status/{upload_id_str}",
            status_code=303)

    except Exception as e:
        # Clean up temp upload dir if saving fails
        if os.path.exists(upload_dir_temp):
             shutil.rmtree(upload_dir_temp)
        # Clean up processed output dir if saving fails
        if os.path.exists(processed_output_dir):
             shutil.rmtree(processed_output_dir)

        error_id = str(uuid.uuid4()) # Use a different ID for errors not added to queue
        task_status[error_id] = {
            "status": "failed",
            "progress": 0.0,
            "logs": [f"Failed to receive or queue file: {e}"],
            "results": [],
            "error": str(e)
        }
        logger.error(f"Failed to receive or queue file: {e}")

        return JSONResponse(
            status_code=500,
            content={"upload_id": error_id, "error_message": str(e)}
        )


# Endpoint to serve the status page
@app.get("/status/{upload_id}", response_class=HTMLResponse)
async def get_status_page(request: Request, upload_id: str):
    # Check if the upload_id exists in task_status or processed_files
    # This ensures that even if the server restarts and task_status is lost,
    # we can potentially show results from processed_files if they exist.
    # For simplicity initially, we will just check task_status.
    if upload_id not in task_status:
         # Or check if processed_files/<upload_id> exists
         if not os.path.exists(os.path.join(PROCESSED_DIR, upload_id)):
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error_message": f"Invalid or expired Upload ID: {upload_id}"},
                status_code=404
            )
            
    return templates.TemplateResponse(
        "status.html", # Need to create status.html
        {"request": request, "upload_id": upload_id}
    )

# Endpoint to get task status and results (for polling)
@app.get("/api/status/{upload_id}")
async def get_task_status(upload_id: str):
    status = task_status.get(upload_id)
    if status:
        return JSONResponse(status)
    
    # If not in task_status, check processed_files directory
    # If found, we might need to reconstruct the results from saved files (e.g., text files)
    # This adds complexity, so for now, just return 404 if not in task_status
    if os.path.exists(os.path.join(PROCESSED_DIR, upload_id)):
         # In a more robust system, you'd load results from disk here
         # For now, we assume task_status holds the current state.
         return JSONResponse(
             status_code=404,
             content={"status": "unknown", "error": "Task status not found, but directory exists. Server may have restarted."}
         )

    return JSONResponse(status_code=404, content={"status": "not_found"})

# Endpoint to get task logs (for polling)
@app.get("/api/logs/{upload_id}")
async def get_task_logs(upload_id: str):
    status = task_status.get(upload_id)
    if status:
        return JSONResponse({"logs": status.get("logs", [])})
    return JSONResponse(status_code=404, content={"logs": []})

# Endpoint to generate audio on demand for a specific box
@app.get("/api/generate_audio/{upload_id}/{box_index}")
async def generate_audio(upload_id: str, box_index: int):
    # Check if the task exists and is completed
    task = task_status.get(upload_id)
    if not task or task["status"] != "completed":
        # Could also check if the processed file exists on disk if persistence is implemented
        return JSONResponse(status_code=404, content={"error": "Task not found or not completed."})

    # Find the specific box result
    box_result = None
    for item in task["results"]:
        if item.get("box_index") == box_index:
            box_result = item
            break

    if not box_result or not box_result.get("text"):
        return JSONResponse(status_code=404, content={"error": "Box not found or no text extracted."})

    cleaned_text = box_result["text"]
    
    # Get the selected language from the task status
    selected_language = task.get("language", "en") # Default to 'en' if not found
    
    # Determine gTTS language code (gTTS uses ISO 639-1 codes)
    # Map selected language codes to gTTS supported codes (e.g., 'en', 'hi', 'te')
    gtts_lang = 'en' # Default to English
    if selected_language == 'hi':
        gtts_lang = 'hi'
    elif selected_language == 'te':
        gtts_lang = 'te'
    # Add more language mappings here if needed
    processed_output_dir = os.path.join(PROCESSED_DIR, upload_id)
    audio_filename = f"output_box_{box_index}.mp3"
    audio_file_path_abs = os.path.join(processed_output_dir, audio_filename)

    # Check if audio already exists to avoid re-generating
    if os.path.exists(audio_file_path_abs):
        audio_path_url = f"/processed_files/{upload_id}/{audio_filename}"
        logger.info(f"Audio already exists for task {upload_id}, box {box_index}. Returning existing.")
        return JSONResponse({"audio_path": audio_path_url})


    try:
        # Generate audio using gTTS
        tts = gTTS(
            text=cleaned_text,
            lang=gtts_lang,
            slow=False # Optional: adjust speed
        )
        tts.save(audio_file_path_abs)
        
        audio_path_url = f"/processed_files/{upload_id}/{audio_filename}"
        logger.info(f"Generated audio for task {upload_id}, box {box_index}.")

        # Update the task_status with the audio path
        box_result["audio_path"] = audio_path_url

        return JSONResponse({"audio_path": audio_path_url})

    except Exception as e:
        logger.error(f"Error generating audio for task {upload_id}, box {box_index}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to generate audio: {e}"})


# Endpoint to summarize text on demand for a specific box
# This will require adding the google-generativeai library and your API key
# Endpoint to summarize text on demand for a specific box
@app.get("/api/summarize/{upload_id}/{box_index}")
async def summarize_text(upload_id: str, box_index: int):
    if gemini_model is None:
        return JSONResponse(status_code=500, content={"error": "Gemini model failed to load."})

    # Check if the task exists and is completed
    task = task_status.get(upload_id)
    if not task or task["status"] != "completed":
        return JSONResponse(status_code=404, content={"error": "Task not found or not completed."})

    # Find the specific box result
    box_result = None
    for item in task["results"]:
        if item.get("box_index") == box_index:
            box_result = item
            break

    if not box_result or not box_result.get("text"):
        return JSONResponse(status_code=404, content={"error": "Box not found or no text extracted."})

    cleaned_text = box_result["text"]

    # Check if summary already exists
    if box_result.get("summary"):
        logger.info(f"Summary already exists for task {upload_id}, box {box_index}. Returning existing.")
        return JSONResponse({"summary": box_result["summary"]})

    try:
        # Generate summary using Gemini API
        prompt = f"Summarize the following text:\n\n{cleaned_text}"
        response = await asyncio.to_thread(gemini_model.generate_content, prompt) # Run blocking call in a thread pool

        summary = response.text.strip()
        logger.info(f"Generated summary for task {upload_id}, box {box_index}.")

        # Update the task_status with the summary
        box_result["summary"] = summary

        return JSONResponse({"summary": summary})

    except Exception as e:
        logger.error(f"Error generating summary for task {upload_id}, box {box_index}: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to generate summary: {e}"})


# Endpoint to list uploaded files
@app.get("/list_uploads/", response_class=HTMLResponse)
async def list_uploads(request: Request):
    # List directories in PROCESSED_DIR, each representing an upload
    upload_ids = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    
    # You might want to get more info about each upload (e.g., upload time, original filename)
    # This would require storing more metadata alongside the processed files.
    # For simplicity now, just list the UUIDs.

    return templates.TemplateResponse(
        "list_uploads.html", # Need to create this template
        {"request": request, "upload_ids": upload_ids}
    )


if __name__ == "__main__":
    import uvicorn
    # Removed reload=True for better performance, can add back for development
    # Note: With reload=True, task_status and the queue would reset on file changes.
    # Better to manage reloading externally during development.
    uvicorn.run("main:app", host="0.0.0.0", port=8008)