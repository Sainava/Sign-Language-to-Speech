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
    A[Raw Sign Video] --> B[Frame Extraction]
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

---

## 🎯 **SIMPLIFIED PRESENTATION WORKFLOW** 
*Perfect for slides and presentations*

### **Training Pipeline - Horizontal Flow**

```mermaid
graph LR
    A[Raw Videos] --> B[YOLOv11<br/>Detection] --> C[MediaPipe<br/>Landmarks] --> D[CNN+LSTM<br/>Model] --> E[Trained<br/>Model]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
```

### **Real-Time Inference - Horizontal Flow**

```mermaid
graph LR
    A[📱 Live Camera] --> B[🌐 WebSocket<br/>Streaming] --> C[🔍 MediaPipe<br/>Processing] --> D[📊 Rolling<br/>Buffer] --> E[🧠 Model<br/>Inference] --> F[💬 Text<br/>Prediction] --> G[🔊 Speech<br/>Output]
    
    style A fill:#e1f5fe
    style B fill:#f0f4c3
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#f3e5f5
    style F fill:#fce4ec
    style G fill:#e0f2f1
```

### **System Architecture - Simplified**

```mermaid
graph LR
    A[Frontend<br/>🖥️ HTML/JS] --> B[WebSocket<br/>🔄 Real-time] --> C[Backend<br/>🖥️ FastAPI] --> D[AI Models<br/>🧠 PyTorch] --> E[Output<br/>🔊 Speech]
    
    style A fill:#e3f2fd
    style B fill:#f1f8e9
    style C fill:#fef7e0
    style D fill:#fce4ec
    style E fill:#e8f5e8
```

### **Data Flow - Core Components**

```mermaid
graph LR
    A[Video<br/>Input] --> B[Person<br/>Detection] --> C[Landmark<br/>Extraction] --> D[Feature<br/>Processing] --> E[Classification] --> F[Speech<br/>Synthesis]
    
    A -.-> A1[Frame<br/>Extraction]
    B -.-> B1[YOLO<br/>Cropping]
    C -.-> C1[Pose + Face<br/>+ Hands]
    D -.-> D1[CNN + LSTM<br/>Branches]
    E -.-> E1[20 Sign<br/>Classes]
    F -.-> F1[Text-to-Speech<br/>API]
    
    style A fill:#bbdefb
    style B fill:#c8e6c9
    style C fill:#dcedc8
    style D fill:#ffe0b2
    style E fill:#f8bbd9
    style F fill:#b2dfdb
```

### **Model Architecture - Essential Flow**

```mermaid
graph LR
    A[Multi-Modal<br/>Input] --> B[Parallel<br/>CNN Branches] --> C[LSTM<br/>Temporal] --> D[Feature<br/>Fusion] --> E[Final<br/>Classification]
    
    A -.-> A1[Pose]
    A -.-> A2[Face] 
    A -.-> A3[Hands]
    
    B -.-> B1[Spatial<br/>Features]
    C -.-> C1[Temporal<br/>Patterns]
    D -.-> D1[Combined<br/>Features]
    E -.-> E1[Sign<br/>Prediction]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
```

---

## 📋 **Quick Reference - Key Points**

| Component | Purpose | Technology |
|-----------|---------|------------|
| **YOLOv11** | Person Detection | Computer Vision |
| **MediaPipe** | Landmark Extraction | Google AI |
| **CNN** | Spatial Feature Learning | Deep Learning |
| **LSTM** | Temporal Sequence Modeling | Recurrent Neural Networks |
| **WebSocket** | Real-time Communication | Web Technology |
| **SpeechSynthesis** | Text-to-Speech | Browser API |

## 🎯 **One-Line Summary**
**Raw sign videos → AI preprocessing → Multi-modal CNN+LSTM → Real-time prediction → Speech output**
