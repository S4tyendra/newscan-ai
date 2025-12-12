# NewScan AI

NewScan AI is a FastAPI-based web application designed to digitize newspaper content. It utilizes computer vision to detect news articles within an image, performs OCR to extract text, and leverages Generative AI to provide summaries and audio conversions.

## Core Capabilities

  * **Object Detection:** Uses a custom trained YOLOv8 model (`best_30_epochs.pt`) to segment specific news articles from full newspaper pages.
  * **OCR Engine:** Implements Tesseract OCR to extract text from the segmented images.
  * **Multi-Language Support:** Supports English (`en`), Hindi (`hi`), and Telugu (`te`).
  * **AI Summarization:** Integrates Google's Gemini 2.0 Flash Lite model to generate concise summaries of extracted articles.
  * **Text-to-Speech:** Uses Google Text-to-Speech (gTTS) to generate audio playback for articles.
  * **Asynchronous Processing:** Handles image processing in background queues to prevent UI blocking.

## Tech Stack

  * **Backend:** Python 3.9+, FastAPI, Uvicorn.
  * **Computer Vision:** Ultralytics YOLOv8, OpenCV.
  * **OCR:** Tesseract-OCR (`pytesseract`).
  * **AI/LLM:** Google Generative AI (`google-generativeai`).
  * **Frontend:** Jinja2 Templates, Vanilla CSS/JS.
  * **Containerization:** Docker.

-----

## Prerequisites

Before running locally, ensure the following are installed:

1.  **Python 3.9+**
2.  **Tesseract OCR Engine:**
      * **Linux:** `sudo apt-get install tesseract-ocr tesseract-ocr-hin tesseract-ocr-tel libtesseract-dev`
      * **Windows:** Download the installer from UB-Mannheim. Add the installation path to your System PATH.
      * **macOS:** `brew install tesseract` (plus language data).
3.  **YOLO Model:** Ensure `best_30_epochs.pt` is placed in the root directory.

-----

## Installation & Usage

### Option 1: Local Deployment

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/s4tyendra/newscan-ai.git
    cd newscan-ai
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration:**

      * Open `main.py`.
      * Ensure `GEMINI_API_KEY` is set correctly.

4.  **Run the Application:**

    ```bash
    python main.py
    ```

    *Alternatively via uvicorn directly:*

    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8008 --reload
    ```

5.  **Access:**
    Open your browser and navigate to `http://localhost:8008`.

### Option 2: Docker Deployment

The project includes a `Dockerfile` pre-configured with Tesseract (English and Hindi) and Python dependencies.

1.  **Build the Image:**

    ```bash
    docker build -t newscan-ai .
    ```

2.  **Run the Container:**

    ```bash
    docker run -p 8008:8008 newscan-ai
    ```

-----

## Project Structure

```text
newscan-ai/
├── best_30_epochs.pt          # Trained YOLO model weights
├── Dockerfile                 # Container configuration
├── main.py                    # Application entry point and API routes
├── requirements.txt           # Python dependencies
├── helpers/
│   └── image_processing.py    # Core logic: YOLO prediction, Crop, OCR
├── static/
│   └── style.css              # Dark mode styling
├── templates/
│   ├── index.html             # Upload interface
│   ├── status.html            # Processing status, results, and modal logic
│   ├── list_uploads.html      # History of processed files
│   └── error.html             # Error handling page
├── uploads/                   # Temp storage for uploads (auto-generated)
└── processed_files/           # Storage for cropped images/audio (auto-generated)
```

## API Endpoints

  * `GET /`: Renders the upload interface.
  * `POST /process/`: Accepts file upload and language selection. Initiates async processing.
  * `GET /api/status/{upload_id}`: Returns JSON status of the processing task.
  * `GET /api/generate_audio/{upload_id}/{box_index}`: Generates MP3 for a specific article.
  * `GET /api/summarize/{upload_id}/{box_index}`: Calls Gemini API to summarize text.

## Known Constraints

  * **Tesseract Path:** If running on Windows locally, you may need to explicitly point `pytesseract.pytesseract.tesseract_cmd` to your executable if it's not in the PATH.
  * **Model Dependency:** The application will fail to start if `best_30_epochs.pt` is missing or corrupt.
  * **API Quotas:** Audio generation and Summarization depend on external Google APIs which may have rate limits.

-----
