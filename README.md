# 🧠 Human Activity Recognition (HAR) using CNN + LSTM

## 📌 Overview
This project implements **Human Activity Recognition (HAR)** using the **UFC dataset**.  
We combine **Convolutional Neural Networks (CNNs)** and **Long Short-Term Memory (LSTM)** networks to recognize human activities from video sequences.

- **CNN** extracts **spatial features** from video frames.
- **LSTM** captures **temporal dependencies** across frames.
- Together, the model learns **motion patterns** for accurate activity recognition.

---

## 📂 Dataset: UFC Dataset
- The **UFC dataset** contains video clips of various human activities.
- Preprocessing steps:
  - Extract frames from each video.
  - Resize frames (e.g., 128×128).
  - Normalize pixel values.
  - Group frames into sequences (e.g., 30 frames per sequence).

---

## ⚙️ Methodology / Workflow

### Steps
1. **Data Preprocessing**
   - Extract and preprocess frames from videos.
   - Generate fixed-length frame sequences.

2. **Feature Extraction (CNN)**
   - A CNN learns **spatial patterns** from each frame.

3. **Temporal Modeling (LSTM)**
   - LSTM processes sequences of CNN features.
   - Captures **time-based activity patterns**.

4. **Classification**
   - Dense + Softmax layer predicts activity labels.

---

### 🔄 Workflow Diagram

flowchart TD

    A[Video Dataset (UFC)] --> B[Frame Extraction & Preprocessing]
    B --> C[CNN Feature Extraction]
    C --> D[LSTM Sequence Modeling]
    D --> E[Dense Layer + Softmax]
    E --> F[Activity Prediction]
    
📊 Model Architecture
CNN Layers → Convolution + MaxPooling

LSTM Layer → Captures sequence patterns

Dense + Softmax → Multi-class classification

🚀 Results
Achieved ~70-75% accuracy on validation set.

Successfully recognizes multiple activities such as:

Walking

Running

Punching

Kicking
(depends on dataset classes)

📁 Repository Structure
bash
Copy code
├── data/               # UFC dataset (preprocessed frames)
├── src/                # Source code
│   ├── preprocess.py   # Frame extraction & preprocessing
│   ├── model.py        # CNN + LSTM architecture
│   ├── train.py        # Training script
│   └── evaluate.py     # Model evaluation
├── notebooks/          # Jupyter notebooks for experiments
├── results/            # Training logs, plots
└── README.md           # Project description
⚡ Installation & Usage
1. Clone the repository
bash
Copy code
git clone https://github.com/your-username/human-activity-recognition.git
cd human-activity-recognition
2. Install dependencies
bash
Copy code
pip install -r requirements.txt
3. Prepare dataset
Download UFC dataset.

Place it inside the data/ folder.

Run preprocessing:

bash
Copy code
python src/preprocess.py
4. Train the model
bash
Copy code
python src/train.py
5. Evaluate the model
bash
Copy code
python src/evaluate.py
🔮 Future Enhancements
✅ Add more activity classes.

✅ Experiment with 3D CNNs and Transformers.

✅ Deploy as a real-time recognition system.

✅ Integrate with mobile/edge devices.

🙌 Acknowledgements
UFC Dataset

TensorFlow / Keras / PyTorch (depending on implementation)

Research papers on CNN+LSTM based HAR
