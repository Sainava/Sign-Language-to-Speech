

## 📹 Sign Language to Speech — Real-Time System

A real-time **Indian Sign Language (ISL)** to **Speech** system that combines **YOLO**, **MediaPipe**, and a custom **CNN+LSTM** model to translate sign language videos into spoken words.

---

###  **Project Overview**

This project enables real-time sign language recognition from webcam video and converts it to spoken text in the browser.

**Key Highlights:**

* Real-time webcam capture in the browser.
* Landmark extraction with **MediaPipe Holistic**.
* **YOLOv11** for robust person detection and frame cropping during training.
* Sequence modeling with a hybrid **CNN + LSTM** PyTorch model.
* Interactive web interface with WebSocket-based streaming.
* Automatic speech output using browser **Text-to-Speech (TTS)**.

---

## ⚙️ **Training Pipeline**

> **How the model was built**

1. **Collect Videos:**

* Raw videos of sign language gestures.

2. **Extract Frames:**

* Videos are split into frame sequences for processing.

3. **YOLOv11 Detection:**

* Each frame is passed through **YOLOv11** to detect and crop the region containing the signer.
* This improves landmark extraction accuracy by focusing only on the signer.

4. **Extract Landmarks:**

* **MediaPipe Holistic** is used to extract:

  * **Pose landmarks**
  * **Face landmarks**
  * **Left & right hand landmarks**

5. **Masking:**

* Binary masks track which landmarks are present/missing per frame.
* This helps the model learn variable-length, partially visible features robustly.

6. **CNN + LSTM Model:**

* **CNN layers** learn spatial features from the landmark sequences.
* **LSTM layers** capture temporal dependencies across frames.
* The final model classifies the sign gesture into one of the predefined sign classes.

---

## 🌐 **Web App Inference Pipeline**

> **How real-time recognition works**

1. **User starts the camera** from the browser.
2. Frames are processed **client-side** with **MediaPipe Holistic** to draw pose, face, and hand landmarks for live feedback.
3. The raw frame is also sent over **WebSocket** to the FastAPI backend.
4. The server:

* Optionally runs YOLO crop (can be skipped).
* Runs **MediaPipe Holistic** again on the cropped/received frame.
* Maintains a **rolling buffer** (`deque`) of landmark sequences.
  5. When the user presses `s`:
* The buffer is **RESET** and inference is **STARTED**.
* When enough frames are collected, the server feeds the landmark sequence through the **CNN + LSTM** model.
  6. The predicted sign word is sent back to the browser in real-time.
  7. The browser displays the word and can **speak it aloud** using the Web Speech API.

---

##  **How to Run**

1. **Install Python dependencies**

```bash
conda create -n isl-speech python=3.9
conda activate isl-speech
pip install -r requirements.txt
```

2. **Place models**

* YOLO weights (`yolo11n.pt`) in `models/`
* Trained PyTorch model (`best_web_model.pth`) in `models/`

3. **Start the FastAPI server**

```bash
uvicorn web_app.app_ws:app --reload
```

4. **Open the Web App**

* Navigate to `http://localhost:8000`
* Click **Start Camera**
* Use `s` to **start inference**, `r` to **reset buffer**

---

##  **Controls**

| Key                  | Action                                                             |
| -------------------- | ------------------------------------------------------------------ |
| `s`                  | Reset buffer and start inference                                   |
| `r`                  | Reset buffer only                                                  |
| **Play Translation** | Click the **Speak Translation** button to hear the translated sign |

---

## 📂 **Project Structure**

```
web_app/
 ├── static/
 │    └── index.html         # Frontend HTML
 ├── app_ws.py               # FastAPI server with WebSocket
 ├── model_def.py            # PyTorch CNN + LSTM model definition
 └── models/
       ├── yolo11n.pt
       └── best_web_model.pth
```

---

## 👥 **Credits**

* **Sainava Modak**
* **Kartik Rajput**




## **Acknowledgements**

* [MediaPipe](https://github.com/google-ai-edge/mediapipe)
* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [PyTorch](https://github.com/pytorch/pytorch)



