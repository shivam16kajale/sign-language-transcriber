# sign-language-transcriber

This project enables real-time **hand sign detection and conversion into text** using a machine learning model trained on hand landmarks captured via MediaPipe.

## Overview

- Detects hand gestures (A-Z, SPACE, BACKSPACE)
- Normalizes hand positions for consistent predictions
- Requires holding a gesture steadily to confirm input
- Displays real-time predictions and builds a live text stream

---

## Project Structure

```
hand-sign-to-text/
│
├── app.py                  # Real-time hand sign detection
├── model_03.pkl            # Trained model file (loaded with pickle)
├── requirements.txt        # Required Python packages
└── hand_signs2.csv         # Optional dataset (for training/debugging)
```

---

##  Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/hand-sign-to-text.git
cd hand-sign-to-text
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python app.py
```

> Press `q` to quit the live video stream.

---

## 🛠 How It Works

1. Uses **MediaPipe Hands** to detect 21 landmarks
2. Normalizes landmarks relative to the wrist (to ignore hand position)
3. Feeds the normalized vector to a trained `model_03.pkl`
4. Uses a timer to confirm gesture stability before accepting input
5. Displays predictions and builds the final output text

### Special Gestures
- `SPACE` → Adds space to the text
- `BACKSPACE` → Removes the last character

---

## 🧠 Model Training (Optional)
If you want to train your own model using `hand_signs2.csv`:
- Each row should contain 63 landmark features and a label (A-Z, SPACE, BACKSPACE)
- Use any classifier (e.g., RandomForest, Optuna-optimized models)

---
![istockphoto-1306688274-612x612](https://github.com/user-attachments/assets/27018e61-3c64-435d-870d-9e7fdeac8175)



## ✅ Example Output
https://www.linkedin.com/posts/shivamkajale_just-completed-a-new-machine-learning-project-activity-7354790103266791424-0KfB?utm_source=share&utm_medium=member_desktop&rcm=ACoAADRLvm8B1FBRAYKjOreoiENIoEZZ1tJtqY0

---

## 📦 Dependencies
```
opencv-python
mediapipe
scikit-learn
numpy
```

---

## ✍️ Author
**Your Name** – shivam kajale [[https://](https://github.com/shivam16kajale)

---

## 📄 License
MIT License. See `LICENSE` for details.

---

Feel free to fork and improve this project, or use it as a base for sign language or gesture control interfaces!
