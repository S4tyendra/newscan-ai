import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import os
import shutil
import uuid
from ultralytics import YOLO # Assuming YOLO is needed for model loading in main
from helpers.image_processing import process_newspaper_image # Import the helper function

# Initialize FastAPI app
app = FastAPI(
    title="Newspaper Text Detection and Recognition",
    description="Web application for detecting and recognizing text in newspaper images",
    version="1.0.0"
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

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Initialize YOLO model (Load once at startup)
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    # Depending on criticality, you might want to raise the exception or handle it differently
    model = None # Set model to None if loading fails

# Root endpoint - serves the upload form
@app.get("/", response_class=HTMLResponse)
def get_upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Process endpoint - handles image upload and processing
@app.post("/process/", response_class=HTMLResponse)
async def process_upload(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form(...) # Get language from form data
):
    if model is None:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error_message": "Image processing model failed to load. Please check server logs."},
            status_code=500
        )

    upload_id = uuid.uuid4()
    upload_dir_temp = os.path.join(UPLOAD_DIR, str(upload_id)) # Use a temp dir for original upload
    processed_output_dir = os.path.join(PROCESSED_DIR, str(upload_id)) # Dir for results

    os.makedirs(upload_dir_temp, exist_ok=True)
    os.makedirs(processed_output_dir, exist_ok=True)

    processed_results = [] # Initialize processed_results

    if file.filename is None:
        raise ValueError("No filename provided for the uploaded file.")
        
    file_path_temp = os.path.join(upload_dir_temp, file.filename)

    try:
        # Save the uploaded file temporarily
        with open(file_path_temp, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the image using the helper function, passing the model instance
        processed_results = process_newspaper_image(file_path_temp, language, processed_output_dir, model)

        # Clean up the temporary uploaded file and its directory
        shutil.rmtree(upload_dir_temp)

        # Render results template
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "processed_items": processed_results,
                "upload_id": upload_id
            }
        )

    except Exception as e:
        # Clean up temp upload dir if processing fails
        if os.path.exists(upload_dir_temp):
             shutil.rmtree(upload_dir_temp)
        # Clean up processed output dir if processing fails partway
        if os.path.exists(processed_output_dir) and not processed_results: # Only remove if no results were generated
             shutil.rmtree(processed_output_dir)
             
        return templates.TemplateResponse(
            "error.html", # You might want an error template
            {"request": request, "error_message": str(e)},
            status_code=500
        )

# Optional: Add an error template
# @app.get("/error")
# async def error_page(request: Request, error_message: str):
#      return templates.TemplateResponse("error.html", {"request": request, "error_message": error_message})


if __name__ == "__main__":
    import uvicorn
    # Removed reload=True for better performance, can add back for development
    uvicorn.run("main:app", host="0.0.0.0", port=8008)