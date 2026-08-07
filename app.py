import os
import base64
import uuid
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_from_directory
from xray_analyzer import XRayAnalyzer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

# Initialize Analyzer
analyzer = XRayAnalyzer()

# Directories
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Helper function to encode image bytes to base64
def get_b64_image(img_np):
    success, encoded_img = cv2.imencode('.jpg', img_np)
    if not success:
        return ""
    b64_string = base64.b64encode(encoded_img).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_string}"

@app.route('/')
def home():
    return render_template('index.html')

# Serve sample images static directory
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/analyze', methods=['POST'])
def analyze_xray():
    try:
        # Check if loading sample or upload
        image_path = None
        temp_file = False
        
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"success": False, "error": "No file selected"}), 400
                
            # Save uploaded file
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)
            temp_file = True
            
        elif 'sample_name' in request.json:
            sample_name = request.json['sample_name']
            # Security check
            safe_name = os.path.basename(sample_name)
            image_path = os.path.join('static', 'samples', safe_name)
            
            if not os.path.exists(image_path):
                return jsonify({"success": False, "error": f"Sample image {safe_name} not found"}), 404
        else:
            return jsonify({"success": False, "error": "No image data provided"}), 400
            
        # Read the original image
        img_np = cv2.imread(image_path)
        if img_np is None:
            return jsonify({"success": False, "error": "Failed to load image. Invalid format."}), 400
            
        # Run AI Analysis and GRAD-CAM
        analysis_result = analyzer.analyze(image_path)
        
        # Extract results
        diagnosis = analysis_result["diagnosis"]
        confidence = analysis_result["confidence"]
        breakdown = analysis_result["breakdown"]
        cam_resized = analysis_result["heatmap"]
        
        # Colorize the heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # Create base64 formats
        original_b64 = get_b64_image(img_np)
        heatmap_b64 = get_b64_image(heatmap_colored)
        
        # Fetch detailed clinical report
        report = analyzer.get_clinical_report(diagnosis, confidence)
        
        # Clean up uploaded file
        if temp_file and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Error removing file {image_path}: {e}")
                
        return jsonify({
            "success": True,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "breakdown": breakdown,
            "original_image": original_b64,
            "heatmap_image": heatmap_b64,
            "raw_grid": analysis_result["raw_grid"],
            "clinical_report": report
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Make sure sample files exist (if not run download first)
    if not os.path.exists(os.path.join('static', 'samples', 'normal.jpg')):
        print("Preloaded samples not found, downloading...")
        import subprocess
        subprocess.run(['python', 'download_samples.py'])
        
    print("Starting Hospital X-ray AI Assistant dev server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
