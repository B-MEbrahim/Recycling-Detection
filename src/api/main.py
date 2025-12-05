from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import io
import base64
from contextlib import asynccontextmanager

from .routes import router
from src.inference.detector import RecyclingDetector

# Global detector instance
detector = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    global detector
    detector = RecyclingDetector(model_path='models/teacher/best.pt')
    print("Detector initialized")
    
    yield
    
    # Shutdown (cleanup if needed)
    print("Shutting down...")

app = FastAPI(
    title="Recycling Detection API",
    description="API for recycling object detection using YOLO",
    version="1.0.0",
    lifespan=lifespan
)

# mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# templates
templates = Jinja2Templates(directory="web/templates")

# include routes
app.include_router(router, prefix='/api/v1')

@app.get("/")
async def home():
    """Serve main page"""
    with open("web/templates/index.html", 'r', encoding='utf-8') as f:  # Added encoding='utf-8'
        return HTMLResponse(content=f.read())
    
@app.websocket("/ws/detect")
async def websocket_detection(websocket: WebSocket):
    """WebSocket for real-time detection"""
    await websocket.accept()

    try:
        while True:
            # recieve image data
            data = await websocket.receive_text()

            # decode base64 image
            header, encode = data.split(",", 1)
            image_data = base64.b64decode(encode)

            # convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # perform detection
            results = detector.detect(image)

            # draw detection
            annoted_image = detector.draw_detections(image, results)

            # convert back to base64
            _, buffer = cv2.imencode('.jpg', annoted_image)
            encoded_result = base64.b64encode(buffer).decode('utf-8')

            # send result
            await websocket.send_text(f"data:image/jpeg;base64,{encoded_result}")
    except Exception as e:
        print(f"Websocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
