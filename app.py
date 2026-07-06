import os
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')

    # Run inference
    results = model.predict(source=img, conf=0.4)
    
    # Extract plot and correct BGR to RGB channel inversion
    res_plotted = results[0].plot()[..., ::-1] 
    result_img = Image.fromarray(res_plotted)

    # Convert output image to base64 for HTML rendering
    buffered = BytesIO()
    result_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    volume_info = ""
    total_volume = 0
    pixel_to_cm = 0.07

    if results[0].boxes:
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

    return jsonify({'image': img_str, 'details': volume_info})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
