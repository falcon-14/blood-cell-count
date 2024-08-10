import torch
from PIL import Image
from flask import Flask, request, render_template, send_file
from io import BytesIO
import base64
import cv2  
import numpy as np

app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt', True)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        img = Image.open(image)
        results = model(img)
        img_np = np.array(img)
        annotated_img = img_np.copy()
        rbc_count = 0
        wbc_count = 0
        platelets_count = 0
        for pred in results.pred[0]:
            label = pred[5]
            conf = pred[4]
            bbox = pred[:4].int().tolist()
            if label == 1.0:  
                color = (0, 0, 255)   
                label_name = "RBC"
                rbc_count += 1
            elif label == 2.0: 
                color = (0, 165, 255) 
                label_name = "WBC"
                wbc_count += 1
            else:
                color = (0, 255, 0) 
                label_name = "Platelets"
                platelets_count += 1
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, f"{label_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        _, img_encoded = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(img_encoded).decode()
        return render_template('result.html', img_base64=img_base64, rbc_count=rbc_count, wbc_count=wbc_count, platelets_count=platelets_count)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)







































