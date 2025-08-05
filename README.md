# 🤟 Sign Language Recognition using MediaPipe & Deep Learning

Real-time Sign Language Recognition using machine learning leverages advanced algorithms to interpret and translate sign language gestures into text or speech instantly. This innovative system enhances communication accessibility for 
the deaf and hard-of-hearing community, facilitating seamless interactions in various environments. A real-time Sign Language Recognition system designed to bridge communication between hearing/speech-impaired individuals and others using computer vision and machine learning.

---

## 🧠 Overview

This project leverages **MediaPipe**, **OpenCV**, and a **Multilayer Perceptron (MLP)** neural network to recognize hand gestures and convert them into readable text in real-time. It is designed to support **10 distinct sign classes** captured from custom datasets.

---

## 📸 Demo

| "I Love You" Sign                    | "Hello" Sign               |
| ------------------------------------ | -------------------------- |
| ![I Love You](assets/i_love_you.png) | ![Hello](assets/hello.png) |

> *Add these screenshots to your repository in the `assets/` folder with the above filenames.*

---

## 📂 Project Structure

* `model/`: Contains trained `.keras` and `.tflite` models
* `data/`: Custom CSV dataset with 42 landmark coordinates per image
* `scripts/`: Python scripts for training and real-time detection
* `utils/`: Utility files (e.g., FPS calculator, helper functions)
* `README.md`: Project documentation

---

## 🧪 Methodology

1. **Data Collection**

   * 10 different signs captured using webcam and MediaPipe
   * Each sign's 21 hand landmarks (x, y) stored in CSV files

2. **Preprocessing**

   * Normalization of 42 features (21 points × 2 coordinates)

3. **Model Training**

   * MLP trained for 1000 epochs using:

     * `Dropout`, `Dense`, and `Softmax` layers
   * Achieved **91% accuracy** on test data

4. **Model Evaluation**

   * Confusion matrix & classification report show high precision

5. **Real-Time Recognition**

   * Live webcam input processed via OpenCV and MediaPipe
   * Output text displayed for each recognized gesture

---

## 🧰 Technologies Used

* **MediaPipe** – Hand landmark detection
* **TensorFlow / Keras** – Model training
* **OpenCV** – Real-time video capture
* **Python (NumPy, Pandas, Matplotlib, Seaborn)** – Data processing and visualization

---

## 📊 Sample Results

**Classification Report**

```
Precision: 0.94  
Recall: 0.91  
F1-Score: 0.91  
Accuracy: 91%
```

---

## 🔮 Future Enhancements

* Support for more signs and dialects (ASL, BSL, ISL)
* Integrate facial expression and body pose for better accuracy
* Text-to-speech synthesis for broader communication
* Mobile and AR/VR deployment using TFLite models
* Multilingual output

---

## 📁 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Real-time Detection

```bash
python scripts/realtime_recognition.py
```

> Make sure your webcam is connected and functioning properly.

---

## 📚 References

* [MediaPipe](https://google.github.io/mediapipe/)
* [TensorFlow](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
* [Scikit-learn](https://scikit-learn.org/)

---

Would you like me to generate this as a downloadable `README.md` file with placeholder image links ready to use?
