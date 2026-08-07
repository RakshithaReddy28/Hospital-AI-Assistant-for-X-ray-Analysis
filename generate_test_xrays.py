import os
import numpy as np
import cv2

# Define target folder for test sheets
TEST_DIR = "test_sheets"
os.makedirs(TEST_DIR, exist_ok=True)

def create_realistic_xray(condition, filename):
    """
    Generates a highly detailed, clean synthetic chest radiograph sheet 
    with clear visual indicators for a specific pathology.
    """
    path = os.path.join(TEST_DIR, filename)
    print(f"Creating X-ray test sheet: {filename} ({condition})")
    
    # 1. Base dark background (512x512)
    img = np.zeros((512, 512), dtype=np.uint8)
    
    # 2. Draw soft-tissue outer bounds (faint gray background chest shape)
    cv2.ellipse(img, (256, 260), (210, 230), 0, 0, 180, 40, -1)
    cv2.ellipse(img, (256, 260), (195, 215), 0, 0, 180, 15, -1)
    
    # 3. Draw central spine vertebrae column & mediastinum (structures of the center)
    cv2.rectangle(img, (244, 40), (268, 480), 75, -1)
    # Vertebrae bands
    for y in range(50, 460, 20):
        cv2.line(img, (244, y), (268, y), 50, 2)
    # Aortic knob / Hilar structures
    cv2.circle(img, (240, 140), 22, 90, -1)
    cv2.ellipse(img, (256, 130), (25, 75), 0, 0, 360, 95, -1)
    
    # 4. Draw Lungs (dark, air-filled cavities)
    # Left Lung cavity
    pts_left = np.array([[220, 95], [238, 115], [238, 385], [145, 425], [95, 385], [75, 195], [135, 105]], np.int32)
    cv2.fillPoly(img, [pts_left], 18)
    
    # Right Lung cavity
    pts_right = np.array([[292, 95], [274, 115], [274, 385], [367, 425], [417, 385], [437, 195], [377, 105]], np.int32)
    cv2.fillPoly(img, [pts_right], 18)
    
    # Gaussian blur to soften lung borders into chest wall
    img = cv2.GaussianBlur(img, (9, 9), 0)
    
    # 5. Draw Clavicles (collar bones at the top)
    cv2.ellipse(img, (165, 85), (85, 12), -12, 0, 360, 85, 4)
    cv2.ellipse(img, (347, 85), (85, 12), 12, 0, 360, 85, 4)
    
    # 6. Draw Rib Cage arches (faint overlay lines)
    for i in range(115, 400, 32):
        cv2.ellipse(img, (155, i), (105, 18), 8, 0, 90, 42, 2)
        cv2.ellipse(img, (357, i), (105, 18), -8, 90, 180, 42, 2)
        
    # 7. Draw diaphragms (bottom curves of lung cavities)
    cv2.ellipse(img, (145, 440), (85, 25), 0, 180, 360, 60, 4)
    cv2.ellipse(img, (367, 440), (85, 25), 0, 180, 360, 60, 4)
    
    # 8. Draw heart (Cardiac Silhouette)
    heart_center = (260, 275)
    heart_axes = (58, 48)
    heart_angle = 15
    heart_color = 95
    
    if condition == "cardiomegaly":
        # Enlarged cardiac shadow extending deep into the left lung zone (CTR > 55%)
        heart_center = (282, 285)
        heart_axes = (112, 78)
        heart_angle = 18
        heart_color = 115
        
    cv2.ellipse(img, heart_center, heart_axes, heart_angle, 0, 360, heart_color, -1)
    
    # 9. Add condition-specific characteristics
    if condition == "normal":
        # Standard vascular lung markings (hilar branching vessels)
        vessel_mask = np.zeros_like(img)
        # Branching lines from left and right hilum
        cv2.line(vessel_mask, (230, 190), (170, 170), 40, 2)
        cv2.line(vessel_mask, (230, 190), (160, 220), 35, 2)
        cv2.line(vessel_mask, (232, 210), (180, 255), 32, 2)
        cv2.line(vessel_mask, (282, 190), (342, 170), 40, 2)
        cv2.line(vessel_mask, (282, 190), (352, 220), 35, 2)
        cv2.line(vessel_mask, (280, 210), (332, 255), 32, 2)
        vessel_mask = cv2.GaussianBlur(vessel_mask, (15, 15), 0)
        img = cv2.addWeighted(img, 1.0, vessel_mask, 0.7, 0)
        
    elif condition == "pneumonia":
        # White cloud-like consolidations inside lung spaces
        pneumo_mask = np.zeros_like(img)
        # Right lung consolidations (middle/lower zone)
        cv2.circle(pneumo_mask, (345, 255), 40, 75, -1)
        cv2.circle(pneumo_mask, (355, 305), 32, 85, -1)
        # Left lung minor patches
        cv2.circle(pneumo_mask, (165, 215), 25, 55, -1)
        pneumo_mask = cv2.GaussianBlur(pneumo_mask, (41, 41), 0)
        img = cv2.addWeighted(img, 1.0, pneumo_mask, 0.75, 0)
        
    elif condition == "effusion":
        # Fluid meniscus accumulating at costophrenic angles (bottom edges)
        fluid_mask = np.zeros_like(img)
        # Right pleural effusion (obscured costophrenic corner)
        pts_fluid = np.array([[274, 340], [330, 330], [390, 345], [422, 420], [274, 420]], np.int32)
        cv2.fillPoly(fluid_mask, [pts_fluid], 115)
        # Left corner minor blunting
        cv2.ellipse(fluid_mask, (125, 425), (75, 45), 0, 0, 360, 95, -1)
        fluid_mask = cv2.GaussianBlur(fluid_mask, (19, 19), 0)
        img = cv2.addWeighted(img, 1.0, fluid_mask, 0.85, 0)
        
    elif condition == "pneumothorax":
        # Visceral pleural line (fine white border) and absence of peripheral vascular markings
        # Let's draw a collapsed lung edge in the upper left lung field
        p_mask = np.zeros_like(img)
        # Fine white margin line outlining collapsed lung border
        pts_collapsed = np.array([[120, 110], [150, 130], [170, 180], [180, 250], [170, 320], [140, 360]], np.int32)
        cv2.polylines(p_mask, [pts_collapsed], False, 120, 2)
        # Make peripheral lung space darker (no vascular markings, pure air)
        cv2.ellipse(p_mask, (100, 170), (25, 60), 0, 0, 360, 40, -1)
        p_mask = cv2.GaussianBlur(p_mask, (5, 5), 0)
        img = cv2.add(img, p_mask)
        
    # 10. Simulate high-frequency camera noise & x-ray film grain texture
    noise = np.random.normal(0, 3.5, img.shape).astype(np.float32)
    img_grainy = img.astype(np.float32) + noise
    img_grainy = np.clip(img_grainy, 0, 255).astype(np.uint8)
    
    # Adjust histogram for radiological contrast levels
    img_radiograph = cv2.equalizeHist(img_grainy)
    img_radiograph = cv2.GaussianBlur(img_radiograph, (3, 3), 0)
    
    # Write label overlays (L and R tags)
    cv2.putText(img_radiograph, "R", (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 180, 2)
    cv2.putText(img_radiograph, "L", (465, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 180, 2)
    
    # Save image
    cv2.imwrite(path, img_radiograph)
    print(f"Saved: {path}")

def main():
    print("Generating custom test sheets...")
    create_realistic_xray("normal", "test_normal_lungs.jpg")
    create_realistic_xray("pneumonia", "test_pneumonia_case.jpg")
    create_realistic_xray("cardiomegaly", "test_cardiomegaly_case.jpg")
    create_realistic_xray("effusion", "test_pleural_effusion.jpg")
    create_realistic_xray("pneumothorax", "test_collapsed_lung.jpg")
    print("All test sheets created successfully.")

if __name__ == "__main__":
    main()
