# Med-X AI: Hospital Assistant for Chest Radiograph Analysis & Explainability

Med-X AI is an interactive, web-based clinical assistant dashboard designed to assist radiologists and healthcare providers by analyzing chest X-ray radiographs and generating explainable AI highlights using **GRAD-CAM** (Gradient-weighted Class Activation Mapping).

The application is built on a high-fidelity hybrid architecture combining deep learning visual feature extraction (**PyTorch/ResNet18**) and image-processing measurements (**OpenCV**) to classify pathologies, construct detailed clinical reports, and output interactive visual guides.

---

## 🌟 Key Features

* **AI Diagnostic Core**: Detects and highlights five major radiological classes:
  * **Normal**: Clear, uniformly ventilated lung fields.
  * **Pneumonia**: Patchy, cloud-like infiltrates and airspace consolidations.
  * **Cardiomegaly**: Enlarged cardiac silhouette (Cardiothoracic Ratio > 50%).
  * **Pleural Effusion**: Obscured/blunted costophrenic angles due to fluid accumulation.
  * **Pneumothorax**: Subtle pleural lines representing partial lung collapse.
* **Explainable AI (GRAD-CAM)**: Backpropagates scores through convolutional neural layers to overlay a colored heatmap (Jet colormap) indicating the exact region the model focused on when making its decision.
* **Three Visualizer Layouts**:
  * **Overlay View**: Blends the heatmap directly onto the X-ray with an adjustable transparency slider.
  * **Split-Screen Wipe**: An interactive slider that lets you drag a vertical divider to compare the raw X-ray vs the AI highlights.
  * **Side-by-Side View**: Compares the raw image and the GRAD-CAM overlay side-by-side.
* **Interactive Pixel Inspector Tooltip**: Hovering your cursor over the radiograph displays the real-time $X, Y$ coordinate and the exact mathematical activation weight ($0.00$ to $1.00$) of that feature cell.
* **Dual-Direction Reports**:
  * **Doctor's Guidelines**: Professional diagnostic observations and recommended clinical next steps (highlighted with amber bullets).
  * **Patient's Self-Care**: Clear precautions regarding rest, hydration, diet, and safety (highlighted with cyan bullets).
* **Export Utilities**: Instantly copy report contents to your clipboard or download a clean print-formatted medical `.txt` report.
* **Graceful Degradation Fallback**: Automatically falls back to a high-fidelity NumPy/OpenCV emulation engine if strict OS security policies block native PyTorch DLL bindings.

---

## 🛠️ Technology Stack

* **Backend**:
  * **Python 3.12 / 3.14**
  * **Flask** (Web server framework)
  * **PyTorch** & **Torchvision** (Deep Learning & GRAD-CAM backpropagation)
  * **OpenCV** (Feature extraction, histogram matching, shapes, and image profiles)
  * **NumPy** & **Pillow** (Array calculations and image handling)
* **Frontend**:
  * **HTML5** & **CSS3** (Custom glassmorphism cards, layouts, and vertical scanline animations)
  * **JavaScript** (ES6+ events, slider math, canvas-cropping, and download blobs)
  * **FontAwesome** & **Google Fonts (Outfit, Inter)**

---

## 📁 Project Structure

```text
├── app.py                     # Flask web server and API endpoints
├── xray_analyzer.py           # PyTorch & OpenCV diagnostic engine with GRAD-CAM Hooks
├── download_samples.py        # Fetches/generates sample clinic X-ray radiographs
├── generate_test_xrays.py     # Creates local high-resolution test radiographs
├── templates/
│   └── index.html             # Main dashboard UI structure
└── static/
    ├── css/
    │   └── style.css          # Medical glassmorphism CSS layout and scanline sweeps
    └── js/
        └── app.js             # Slider wipe controllers, mouse coordinate mapping, and export blobs
```

---

## 🚀 How to Install and Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/hospital-xray-ai-assistant.git
cd hospital-xray-ai-assistant
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```
*(Dependencies: `flask`, `torch`, `torchvision`, `numpy`, `opencv-python`, `pillow`)*

### 3. Initialize Samples (Optional)
Run the downloader utility to set up the default sample chest radiographs for the gallery:
```bash
python download_samples.py
```

### 4. Launch the Web Application
Start the Flask local development server:
```bash
python app.py
```

Open your browser and navigate to:
👉 https://rakshithareddy28.pythonanywhere.com

---

## ⚠️ Clinical Notice & Disclaimer
This project is developed for **educational, research, and auxiliary demonstration purposes only**. All generated reports, diagnoses, and GRAD-CAM visualizations are produced by experimental code and should not be used as clinical recommendations, official medical device findings, or diagnosis reports. Treat this as an assistant study tool. Always consult a board-certified radiologist for medical diagnostic determinations.
