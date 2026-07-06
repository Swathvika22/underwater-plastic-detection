import base64
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Load the trained weights
model = YOLO('best.pt')

# Modern, Deep-Sea Themed Frontend UI
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pelagic Vision | Marine Plastic Segmentation</title>
    <style>
        body {
            background: linear-gradient(135deg, #020b14 0%, #061a30 100%);
            color: #e0f2fe;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            margin-top: 60px;
            background: rgba(10, 37, 64, 0.6);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 16px;
            padding: 40px;
            width: 85%;
            max-width: 1000px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }
        h1 {
            text-align: center;
            color: #38bdf8;
            font-weight: 300;
            letter-spacing: 2px;
            margin-bottom: 5px;
        }
        .subtitle {
            text-align: center;
            color: #7dd3fc;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        .upload-area {
            border: 2px dashed #0ea5e9;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
            background: rgba(14, 165, 233, 0.05);
        }
        .upload-area:hover {
            background: rgba(14, 165, 233, 0.15);
        }
        #file-input { display: none; }
        button {
            background: #0ea5e9;
            color: #020b14;
            border: none;
            padding: 14px 24px;
            border-radius: 6px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            transition: background 0.3s;
        }
        button:hover { background: #38bdf8; }
        .results-grid {
            display: none;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-top: 30px;
        }
        .image-box img {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #0ea5e9;
        }
        .info-box {
            background: rgba(2, 11, 20, 0.5);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid rgba(56, 189, 248, 0.2);
        }
        .info-box h3 {
            color: #38bdf8;
            margin-top: 0;
            border-bottom: 1px solid #0ea5e9;
            padding-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PELAGIC VISION</h1>
        <div class="subtitle">Underwater Instance Segmentation & Volumetric Analysis</div>
        
        <div class="upload-area" onclick="document.getElementById('file-input').click()">
            <p>Click or drag an image here to initiate scan</p>
            <input type="file" id="file-input" accept="image/*" onchange="updateFileName()">
            <p id="file-name" style="color: #38bdf8; font-size: 0.85em;"></p>
        </div>
        
        <button id="submit-btn" onclick="uploadImage()">Run Neural Network</button>

        <div class="results-grid" id="results">
            <div class="image-box">
                <img id="output-image" src="" alt="Detection Output">
            </div>
            <div class="info-box">
                <h3>Analysis Results</h3>
                <div id="volume-data" style="line-height: 1.6;"></div>
            </div>
        </div>
    </div>

    <script>
        function updateFileName() {
            const input = document.getElementById('file-input');
            if (input.files.length > 0) {
                document.getElementById('file-name').innerText = "Selected: " + input.files[0].name;
            }
        }

        async function uploadImage() {
            const fileInput = document.getElementById('file-input');
            if (fileInput.files.length === 0) return alert('Select an image prior to execution.');
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const btn = document.getElementById('submit-btn');
            btn.innerText = 'Processing Tensors...';
            btn.style.background = '#7dd3fc';

            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();

                document.getElementById('output-image').src = 'data:image/jpeg;base64,' + data.image;
                document.getElementById('volume-data').innerHTML = data.details;
                document.getElementById('results').style.display = 'grid';
            } catch (error) {
                alert('Inference failed. Check server logs.');
            } finally {
                btn.innerText = 'Run Neural Network';
                btn.style.background = '#0ea5e9';
            }
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(content=HTML_CONTENT)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')

    # Lowered confidence threshold from 0.4 to 0.25 to catch more plastic
    results = model.predict(source=img, conf=0.25)
    
    # Correct BGR to RGB channel inversion!
    res_plotted = results[0].plot()[..., ::-1] 
    result_img = Image.fromarray(res_plotted)

    buffered = io.BytesIO()
    result_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    volume_info = ""
    total_volume = 0
    pixel_to_cm = 0.07

    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            label = model.names[int(box.cls)]
            w_px = box.xywh[0][2].item()
            h_px = box.xywh[0][3].item()
            
            w_cm = w_px * pixel_to_cm
            h_cm = h_px * pixel_to_cm
            obj_volume = 3.14159 * (w_cm / 2) * (h_cm / 2) * ((w_cm + h_cm) / 4)
            
            total_volume += obj_volume
            volume_info += f"Item: {label} | Estimated Volume: {obj_volume:.2f} cm³<br>"
        
        volume_info += f"<br><strong>--- TOTAL VOLUME: {total_volume:.2f} cm³ ---</strong>"
    else:
        volume_info = "No plastic detected."

    return {"image": img_str, "details": volume_info}
