import os
import requests
import numpy as np
import cv2

# Define target samples directory
SAMPLES_DIR = os.path.join("static", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# List of Wikipedia public-domain chest X-rays
SAMPLE_URLS = {
    "normal.jpg": "https://upload.wikimedia.org/wikipedia/commons/a/a1/Normal_posteroanterior_chest_radiograph.jpg",
    "pneumonia.jpg": "https://upload.wikimedia.org/wikipedia/commons/e/e6/Pneumonia_on_Chest_X-Ray.jpg",
    "cardiomegaly.jpg": "https://upload.wikimedia.org/wikipedia/commons/f/fb/Cardiomegaly.jpg",
    "effusion.jpg": "https://upload.wikimedia.org/wikipedia/commons/a/a3/Pleural_effusion_y_atelectasia.jpg"
}

def generate_synthetic_xray(path, condition):
    """
    Generates a beautiful synthetic chest X-ray image using OpenCV drawing functions
    as a robust fallback when internet connection is offline or slow.
    """
    print(f"Generating synthetic X-ray for condition: {condition} -> {path}")
    
    # 1. Base dark background
    img = np.zeros((512, 512), dtype=np.uint8)
    
    # 2. Draw chest wall/rib cage outline (soft grey)
    cv2.ellipse(img, (256, 250), (200, 220), 0, 0, 180, 50, -1)
    cv2.ellipse(img, (256, 250), (190, 210), 0, 0, 180, 20, -1)
    
    # 3. Draw mediastinum & spine (center column)
    cv2.rectangle(img, (246, 50), (266, 470), 80, -1)
    cv2.ellipse(img, (256, 120), (25, 80), 0, 0, 360, 95, -1)
    
    # 4. Draw Lungs (dark regions inside rib cage)
    # Left Lung
    pts_left = np.array([[220, 100], [240, 120], [240, 380], [150, 420], [100, 380], [80, 200], [140, 110]], np.int32)
    cv2.fillPoly(img, [pts_left], 20)
    
    # Right Lung
    pts_right = np.array([[292, 100], [272, 120], [272, 380], [362, 420], [412, 380], [432, 200], [372, 110]], np.int32)
    cv2.fillPoly(img, [pts_right], 20)
    
    # Smooth the lungs outline a bit
    img = cv2.GaussianBlur(img, (7, 7), 0)
    
    # 5. Draw clavicles (collarbones)
    cv2.ellipse(img, (170, 90), (90, 15), -15, 0, 360, 90, 3)
    cv2.ellipse(img, (342, 90), (90, 15), 15, 0, 360, 90, 3)
    
    # 6. Draw ribs (horizontal bands)
    for i in range(120, 400, 30):
        # Left rib curves
        cv2.ellipse(img, (160, i), (100, 20), 10, 0, 90, 45, 1)
        # Right rib curves
        cv2.ellipse(img, (352, i), (100, 20), -10, 90, 180, 45, 1)
        
    # 7. Draw cardiac silhouette (Heart)
    # Default parameters
    heart_center = (260, 280)
    heart_axes = (60, 50)
    heart_angle = 15
    
    if condition == "cardiomegaly":
        # Make the heart significantly larger, extending to the left lung field
        heart_center = (280, 290)
        heart_axes = (110, 80)
        heart_angle = 20
        heart_color = 120
    else:
        heart_color = 100
        
    cv2.ellipse(img, heart_center, heart_axes, heart_angle, 0, 360, heart_color, -1)
    
    # 8. Apply condition-specific features
    if condition == "pneumonia":
        # Draw patchy consolidations (blurry light spots in lung fields)
        # Right lung consolidation
        patch1 = np.zeros_like(img)
        cv2.circle(patch1, (340, 260), 45, 80, -1)
        cv2.circle(patch1, (350, 310), 35, 90, -1)
        patch1 = cv2.GaussianBlur(patch1, (45, 45), 0)
        img = cv2.addWeighted(img, 1.0, patch1, 0.7, 0)
        
        # Left lung minor consolidation
        patch2 = np.zeros_like(img)
        cv2.circle(patch2, (160, 220), 30, 60, -1)
        patch2 = cv2.GaussianBlur(patch2, (25, 25), 0)
        img = cv2.addWeighted(img, 1.0, patch2, 0.5, 0)
        
    elif condition == "effusion":
        # Draw fluid line covering the bottom corners (blunting costophrenic angles)
        fluid_mask = np.zeros_like(img)
        # Left side fluid (costophrenic angle blunting)
        cv2.ellipse(fluid_mask, (130, 430), (80, 50), 0, 0, 360, 110, -1)
        # Right side high fluid level
        pts_fluid_r = np.array([[272, 350], [320, 340], [380, 350], [420, 420], [272, 420]], np.int32)
        cv2.fillPoly(fluid_mask, [pts_fluid_r], 120)
        fluid_mask = cv2.GaussianBlur(fluid_mask, (15, 15), 0)
        img = cv2.addWeighted(img, 1.0, fluid_mask, 0.8, 0)
        
    elif condition == "normal":
        # Keep it clean, add standard soft vascular markings
        vascular = np.zeros_like(img)
        # Left hilar region
        cv2.line(vascular, (230, 200), (180, 180), 45, 2)
        cv2.line(vascular, (230, 200), (160, 230), 40, 2)
        cv2.line(vascular, (230, 220), (170, 260), 35, 2)
        # Right hilar region
        cv2.line(vascular, (280, 200), (330, 180), 45, 2)
        cv2.line(vascular, (280, 200), (350, 230), 40, 2)
        cv2.line(vascular, (280, 220), (340, 260), 35, 2)
        vascular = cv2.GaussianBlur(vascular, (11, 11), 0)
        img = cv2.addWeighted(img, 1.0, vascular, 0.6, 0)
        
    # 9. Apply noise and texture for clinical realism
    noise = np.random.normal(0, 3, img.shape).astype(np.float32)
    img_float = img.astype(np.float32) + noise
    img_float = np.clip(img_float, 0, 255).astype(np.uint8)
    
    # Apply standard medical look histogram adjustment
    img_final = cv2.equalizeHist(img_float)
    img_final = cv2.GaussianBlur(img_final, (3, 3), 0)
    
    # Save image
    cv2.imwrite(path, img_final)

def download_samples():
    print("Initiating X-ray sample retrieval...")
    for filename, url in SAMPLE_URLS.items():
        filepath = os.path.join(SAMPLES_DIR, filename)
        condition = filename.split(".")[0]
        
        try:
            print(f"Downloading {filename} from {url}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                # Verify that the downloaded file is a valid image
                test_img = cv2.imread(filepath)
                if test_img is None or test_img.size == 0:
                    raise ValueError("Downloaded file is not a valid image")
                print(f"Successfully downloaded and verified {filename}")
            else:
                raise Exception(f"HTTP response status code {response.status_code}")
        except Exception as e:
            print(f"Failed to download {filename} ({e}). Falling back to synthetic generation.")
            generate_synthetic_xray(filepath, condition)

if __name__ == "__main__":
    download_samples()
    print("X-ray sample initialization complete.")
