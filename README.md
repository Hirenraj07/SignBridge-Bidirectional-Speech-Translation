# SignBridge-Bidirectional-Speech-Translation
#  SignBridge: ANN-based Indian Sign Language Recognition
SignBridge is an open-source project that enables **Indian Sign Language (ISL) recognition** using a **Convolutional Neural Network (CNN)** pipeline. It processes hand-sign images and converts them into meaningful text predictions.

---
##  Key Features
- ✔️ Custom-trained **CNN classifier** for ISL  
- ✔️ Clean training → evaluation → inference pipeline  
- ✔️ Modular code structure (`src/`)  
- ✔️ Pretrained weights support (`models/`)  
- ✔️ API-ready demo app (`src/app.py`)  
- ✔️ Example predictions & screenshots

  ## 📊 Dataset
This project uses the **WSASL (Word-Level Sign Language) Dataset** from **Kaggle**.

🔗 Kaggle Dataset Link:  
[https://www.kaggle.com/datasets/grassknoted/word-level-sign-language-dataset](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)

WLASL is the largest video dataset for Word-Level American Sign Language (ASL) recognition, which features 2,000 common different words in ASL. We hope WLASL will facilitate the research in sign language understanding and eventually benefit the communication between deaf and hearing communities.
Download the dataset from Kaggle and extract it into the following structure:
import kagglehub

# Download latest version
path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")

print("Path to dataset files:", path)

Create environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

install dependencies
pip install -r requirements.txt

MediaPipe + CNN Classifier + Real-time Hand Landmark Extraction

**Implementation**:

**MediaPipe Hands**

Used to extract 21 key hand landmarks (x, y coordinates).

**Landmark Preprocessing**

Normalized coordinates

Flattened into a feature vector of length 42

Stored for training

Classifier Model

You used:

A Sequential ANN / small CNN-like dense network

Input: 42 features

Dense layers

Softmax output over classes

NOT a heavy CNN on images —
We Did not pass images directly into CNN.
We passed MediaPipe landmark features into a neural network.

**Training**

We trained on WSASL Kaggle dataset converted to MediaPipe landmark arrays.

**Inference Pipeline**

Read image/video frame

Extract hand landmarks

Send normalized 42-dim vector → model

Predict label

Display output

**Demo App**

app.py used webcam / image upload → MediaPipe → model → prediction

python src/evaluate.py --weights models/sign_language_model.h5

**Real Time Demo:**
python src/app.py --weights models/sign_language_model.h5



