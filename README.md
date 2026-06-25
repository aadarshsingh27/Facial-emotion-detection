# FacePredictor

Webcam face analysis project for predicting **gender**, **age**, and **emotion** using PyTorch models with:
- a **Streamlit app** (`app/FacePredictor_app.py`)
- a **Jupyter notebook app** (`app/FacePredictor_app.ipynb`)

---

## Features

- Face detection using **MediaPipe**
- Gender prediction using **EfficientNet-B0**
- Age estimation using **VGG19** (regression)
- Emotion classification using **VGG19** (7 classes)
- Annotated prediction output on detected faces

---

## Model Weights (Required)

The trained model files are not committed to this repository due to size.

Download and place these files inside the `app/` directory:

- `fine_tuned_gender_model.pth`
- `fine_tuned_age_model.pth`
- `emotion_model_fine_tuned.pth`

Download links:
- Gender: [Google Drive](https://drive.google.com/drive/folders/12GnZIXySo3p3KtZfxs9XUkd-WLXI7fAh?usp=sharing)
- Age: [Google Drive](https://drive.google.com/drive/folders/1z3bun3mhV9AAC0FsfDibIX0WnS2h3GIu?usp=sharing)
- Emotion: [Google Drive](https://drive.google.com/drive/folders/1AcNPYxRBX-YTpB38tjNcYQ73xyn66jN6?usp=sharing)

---

## Project Structure

```text
Facialemotion-det/
├── app/
│   ├── FacePredictor_app.py
│   ├── FacePredictor_app.ipynb
│   ├── fine_tuned_gender_model.pth
│   ├── fine_tuned_age_model.pth
│   └── emotion_model_fine_tuned.pth
├── notebooks/
│   ├── gender_model_notebook.ipynb
│   ├── age_model_notebook.ipynb
│   └── emotion_model_notebook.ipynb
├── requirements.txt
└── README.md
```

> `notebooks/` contains training/fine-tuning notebooks and is optional for inference-only usage.

---

## Setup

### 1) Open project folder

```bash
cd Facialemotion-det
```

### 2) (Recommended) Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

If you plan to run Jupyter notebooks, ensure Jupyter is installed:

```bash
pip install jupyter ipykernel
```

---

## Run with Streamlit

From project root:

```bash
streamlit run app/FacePredictor_app.py
```

How it works:
- Click camera capture in the app
- The image is processed for face detection
- Predictions are displayed for gender, age, and emotion

---

## Run with Jupyter Notebook

From project root:

```bash
jupyter notebook
```

Then:
1. Open `app/FacePredictor_app.ipynb`
2. Select the correct Python kernel (same environment as installed packages)
3. Run cells from top to bottom
4. Allow camera permissions when prompted

---

## Models and Datasets

- **Gender model**
  - Backbone: EfficientNet-B0
  - Dataset: [UTKFace](https://susanqq.github.io/UTKFace/)

- **Age model**
  - Backbone: VGG19
  - Task: Regression (predict age)
  - Dataset: [UTKFace](https://susanqq.github.io/UTKFace/)

- **Emotion model**
  - Backbone: VGG19
  - Classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
  - Dataset: [FER2013 (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)

---

## Troubleshooting

- **`FileNotFoundError` for model weights**
  - Confirm all `.pth` files are inside `app/`.

- **`ModuleNotFoundError`**
  - Reinstall dependencies: `pip install -r requirements.txt`
  - Verify notebook kernel points to the same Python environment.

- **Webcam not opening**
  - Close other apps using webcam (Zoom/Teams/browser tabs).
  - Grant camera permission to browser/Jupyter/Streamlit.

---

## Acknowledgments

- PyTorch and TorchVision pretrained backbones (EfficientNet-B0, VGG19)
- MediaPipe for face detection
- Streamlit for web UI

