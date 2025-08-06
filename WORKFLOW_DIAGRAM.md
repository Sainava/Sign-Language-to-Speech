# 🔄 Sign Language to Speech - Complete Workflow Diagram

## 📊 **TRAINING PIPELINE**

```mermaid
graph TD
    A[Raw Sign Language Videos] --> B[Frame Extraction]
    B --> C[YOLOv11 Person Detection]
    C --> D[Crop Person Regions]
    D --> E[MediaPipe Holistic Processing]
    E --> F[Extract Pose Landmarks]
    E --> G[Extract Face Landmarks]
    E --> H[Extract Left Hand Landmarks]
    E --> I[Extract Right Hand Landmarks]
    
    F --> J[Create Pose Masks]
    G --> K[Create Face Masks]
    H --> L[Create Left Hand Masks]
    I --> M[Create Right Hand Masks]
    
    J --> N[Normalize Landmarks]
    K --> N
    L --> N
    M --> N
    
    N --> O[Sequence Formation]
    O --> P[Data Augmentation]
    P --> Q[Training Dataset]
    
    Q --> R[CNN Branch - Pose]
    Q --> S[CNN Branch - Face]
    Q --> T[CNN Branch - Left Hand]
    Q --> U[CNN Branch - Right Hand]
    
    R --> V[LSTM Processing - Pose]
    S --> W[LSTM Processing - Face]
    T --> X[LSTM Processing - Left Hand]
    U --> Y[LSTM Processing - Right Hand]
    
    V --> Z[Feature Fusion]
    W --> Z
    X --> Z
    Y --> Z
    
    Z --> AA[Classifier Head]
    AA --> BB[Cross Entropy Loss]
    BB --> CC[Backpropagation]
    CC --> DD[Model Optimization]
    DD --> EE[Trained Model Weights]
```

## 🌐 **REAL-TIME INFERENCE PIPELINE**

```mermaid
graph TD
    A[User Opens Web Browser] --> B[Load HTML Interface]
    B --> C[Start Camera Button Clicked]
    C --> D[WebRTC Camera Access]
    D --> E[MediaPipe Holistic - Client Side]
    E --> F[Draw Landmarks on Canvas]
    F --> G[Capture Frame to Base64]
    G --> H[Send Frame via WebSocket]
    
    H --> I[FastAPI WebSocket Server]
    I --> J[Decode Base64 Frame]
    J --> K[Optional: YOLOv11 Cropping]
    K --> L[MediaPipe Holistic - Server Side]
    
    L --> M[Extract Pose Landmarks]
    L --> N[Extract Face Landmarks]
    L --> O[Extract Left Hand Landmarks]
    L --> P[Extract Right Hand Landmarks]
    
    M --> Q[Generate Pose Mask]
    N --> R[Generate Face Mask]
    O --> S[Generate Left Hand Mask]
    P --> T[Generate Right Hand Mask]
    
    Q --> U[Rolling Buffer - Pose Sequence]
    R --> V[Rolling Buffer - Face Sequence]
    S --> W[Rolling Buffer - Left Hand Sequence]
    T --> X[Rolling Buffer - Right Hand Sequence]
    
    U --> Y{Buffer Full?}
    V --> Y
    W --> Y
    X --> Y
    
    Y -->|No| Z[Continue Buffering]
    Z --> H
    
    Y -->|Yes + Inference Triggered| AA[CNN Branch Processing]
    AA --> BB[LSTM Temporal Modeling]
    BB --> CC[Feature Concatenation]
    CC --> DD[Classifier Prediction]
    DD --> EE[Apply Confidence Threshold]
    EE --> FF[Generate Prediction]
    FF --> GG[Send Result via WebSocket]
    
    GG --> HH[Display Prediction on UI]
    HH --> II{Speech Enabled?}
    II -->|Yes| JJ[SpeechSynthesis API]
    II -->|No| KK[Silent Display]
    JJ --> LL[Audio Output]
    KK --> MM[Continue Loop]
    LL --> MM
    MM --> H
```

## 🧠 **DETAILED MODEL ARCHITECTURE FLOW**

```mermaid
graph TD
    A[Input: Multi-Modal Landmark Sequences] --> B[Pose Branch Input]
    A --> C[Face Branch Input]
    A --> D[Left Hand Branch Input]
    A --> E[Right Hand Branch Input]
    
    B --> F[Pose CNN Layers]
    C --> G[Face CNN Layers]
    D --> H[Left Hand CNN Layers]
    E --> I[Right Hand CNN Layers]
    
    F --> J[Conv1D + BatchNorm + ReLU + Dropout]
    G --> K[Conv1D + BatchNorm + ReLU + Dropout]
    H --> L[Conv1D + BatchNorm + ReLU + Dropout]
    I --> M[Conv1D + BatchNorm + ReLU + Dropout]
    
    J --> N[Global Mean Pooling - Landmarks]
    K --> O[Global Mean Pooling - Landmarks]
    L --> P[Global Mean Pooling - Landmarks]
    M --> Q[Global Mean Pooling - Landmarks]
    
    N --> R[Apply Pose Mask]
    O --> S[Apply Face Mask]
    P --> T[Apply Left Hand Mask]
    Q --> U[Apply Right Hand Mask]
    
    R --> V[LSTM Temporal Processing]
    S --> W[LSTM Temporal Processing]
    T --> X[LSTM Temporal Processing]
    U --> Y[LSTM Temporal Processing]
    
    V --> Z[Masked Average Pooling]
    W --> AA[Masked Average Pooling]
    X --> BB[Masked Average Pooling]
    Y --> CC[Masked Average Pooling]
    
    Z --> DD[Feature Concatenation]
    AA --> DD
    BB --> DD
    CC --> DD
    
    DD --> EE[Linear Layer 1]
    EE --> FF[ReLU Activation]
    FF --> GG[Dropout]
    GG --> HH[Linear Layer 2]
    HH --> II[Final Logits]
    II --> JJ[Softmax Probabilities]
    JJ --> KK[Predicted Class]
```

## 🔧 **DATA PREPROCESSING WORKFLOW**

```mermaid
graph TD
    A[Raw Videos] --> B[Extract Frames Script]
    B --> C[Frame Sequences]
    C --> D[YOLO Detection Script]
    D --> E[Cropped Person Frames]
    E --> F[Landmark Extraction Script]
    F --> G[Raw Landmark Arrays]
    
    G --> H[Pose Landmarks]
    G --> I[Face Landmarks]
    G --> J[Left Hand Landmarks]
    G --> K[Right Hand Landmarks]
    
    H --> L[Mask Generation Script]
    I --> L
    J --> L
    K --> L
    
    L --> M[Binary Validity Masks]
    M --> N[Normalization Script]
    N --> O[Normalized Landmarks]
    O --> P[Training Ready Dataset]
    
    P --> Q[Train/Validation Split]
    Q --> R[DataLoader Creation]
    R --> S[Batch Processing]
    S --> T[Model Training Loop]
```

## 🎮 **USER INTERACTION FLOW**

```mermaid
graph TD
    A[User Loads Web Page] --> B[Camera Permission Request]
    B --> C[MediaPipe Initialization]
    C --> D[WebSocket Connection]
    D --> E[Live Camera Feed Active]
    
    E --> F[User Presses 'S' Key]
    F --> G[Reset Buffer + Start Inference]
    G --> H[Continuous Frame Processing]
    
    H --> I[Sign Detection]
    I --> J[Display Prediction]
    J --> K{Speech Enabled?}
    K -->|Yes| L[Automatic TTS]
    K -->|No| M[Silent Mode]
    
    L --> N[Audio Feedback]
    M --> N
    N --> O[Continue Monitoring]
    
    E --> P[User Presses 'R' Key]
    P --> Q[Reset Buffer Only]
    Q --> O
    
    E --> R[User Toggles Speech Button]
    R --> S[Enable/Disable TTS]
    S --> O
    
    E --> T[User Toggles Landmarks]
    T --> U[Show/Hide Visual Landmarks]
    U --> O
    
    O --> H
```

## 📱 **SYSTEM COMPONENT INTERACTION**

```mermaid
graph TD
    A[Frontend - HTML/JS] --> B[MediaPipe JavaScript]
    A --> C[WebSocket Client]
    A --> D[SpeechSynthesis API]
    A --> E[Canvas Rendering]
    
    C --> F[FastAPI WebSocket Server]
    F --> G[MediaPipe Python]
    F --> H[PyTorch Model]
    F --> I[YOLO Model]
    
    G --> J[Landmark Extraction]
    H --> K[CNN+LSTM Inference]
    I --> L[Person Detection]
    
    J --> M[Feature Processing]
    K --> N[Prediction Generation]
    L --> O[Frame Cropping]
    
    M --> F
    N --> F
    O --> F
    
    F --> C
    C --> A
```

## 🔄 **COMPLETE END-TO-END FLOW**

```mermaid
graph TD
    A[🎥 Raw Sign Video] --> B[📹 Frame Extraction]
    B --> C[YOLOv11 Detection]
    C --> D[Person Cropping]
    D --> E[MediaPipe Holistic]
    
    E --> F[Pose Landmarks]
    E --> G[Face Landmarks]
    E --> H[Left Hand Landmarks]
    E --> I[Right Hand Landmarks]
    
    F --> J[CNN Feature Extraction]
    G --> J
    H --> J
    I --> J
    
    J --> K[LSTM Temporal Modeling]
    K --> L[Feature Fusion]
    L --> M[Classification]
    M --> N[Confidence Scoring]
    N --> O[Text Prediction]
    O --> P[Speech Synthesis]
    P --> Q[Audio Output]
```

---

## 📋 **Key Workflow Stages Summary**

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| **Data Collection** | Raw videos | Manual recording | Video files |
| **Preprocessing** | Videos | Frame extraction → YOLO → MediaPipe | Landmark sequences |
| **Training** | Landmark data | CNN+LSTM training | Trained model |
| **Deployment** | Trained model | FastAPI server | Web service |
| **Inference** | Live camera | Real-time processing | Sign predictions |
| **User Experience** | Predictions | Display + TTS | Audio/visual feedback |

---

## 🎯 **Critical Decision Points**

1. **YOLO Detection**: Person present? → Crop or use full frame
2. **MediaPipe Processing**: Landmarks detected? → Generate masks
3. **Buffer Management**: Sequence complete? → Trigger inference
4. **Confidence Threshold**: Prediction reliable? → Display or ignore
5. **Speech Control**: Audio enabled? → Speak or stay silent

This workflow represents your complete Sign Language to Speech pipeline from raw data to final user interaction! 🚀
