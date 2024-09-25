# 🌱 Dr. Greenthumb - Decoding Nature's Needs

**Team Members**  
Sonia Mayel | Sadia Khan Rupa | Marvin Lipps

## 📝 Overview
**Dr. Greenthumb** is an AI-powered assistant designed to diagnose plant diseases and provide solutions. This 3-week final project of my Data Science Bootcamp combines **deep learning**, **LLMs**, and an intuitive **Streamlit** interface to offer real-time plant care solutions, including plant disease detection via computer vision and an integrated chatbot for further guidance.

---

## 🚀 Features

### 🌿 Plant Disease Classification
- Detects common tomato diseases using a **TensorFlow/Keras** model with **98% accuracy**.
- Supports image uploads via file or webcam for real-time disease prediction.

### 💬 AI Chatbot with LLM
- Conversational AI built on top of a fine-tuned **LLM** for expert plant care advice.
- Utilizes **Retrieval-Augmented Generation (RAG)** to fetch disease treatment information.

### 🖥️ User Interface with Streamlit
- **Streamlit-powered** app for seamless interaction.
- Allows users to upload images or use a webcam to capture plant conditions, providing instant disease detection, treatment recommendations from the chatbot, and answers to further questions.

---

## 📊 Machine Learning & Technical Overview

### 🔬 Image Classification Model
- **CNN** fine-tuned to classify 10 types of tomato diseases.
- Trained on a dataset of **18,000 images** with **data augmentation** techniques to improve model performance.
- **Performance Metrics**:  
  - Accuracy: **98%**  
  - Precision: **98.84%**  
  - F1-Score: **98.57%**

### 🤖 LLM Chatbot Integration
- Built using **Mistral 7B Instruct** and **Llama-3**, fine-tuned for plant-specific knowledge.
- Integrates with **LangChain** and **Chroma** for accurate responses based on plant disease queries.

---

## 💻 Tech Stack & Libraries Used
- **Backend**: Python, TensorFlow, Keras, LangChain, LlamaCpp
- **Frontend**: Streamlit, OpenCV (for image handling)
- **Libraries**:
  - TensorFlow/Keras: Deep learning and image classification.
  - LangChain: Conversational logic for the chatbot.
  - Chroma: Vector database for RAG integration.
  - Streamlit: Frontend and app deployment.
  - OpenCV: Webcam and image processing.
  - HuggingFace: Embeddings for chatbot fine-tuning.
  - Numpy, Matplotlib, Seaborn, RegEx

---

## 📂 Dataset and Augmentation
- Trained on **18,000 images** across 10 tomato disease categories, including healthy conditions.
- **Data Augmentation** techniques (e.g., rotation, zoom) applied to enhance model generalization.
  
**Detected Diseases**:
1. Bacterial spots  
2. Early blight  
3. Healthy  
4. Late blight  
5. Leaf curl  
6. Leaf mold  
7. Mosaic virus  
8. Septoria leaf spots  
9. Spider mites  
10. Target spots

---

## 🖼️ How It Works

1. **Image Upload**: Users upload an image or use the webcam.
2. **Disease Detection**: The CNN model predicts the disease, providing confidence scores.
3. **Conversational Assistance**: The chatbot provides treatment advice based on the detected disease.

---

## 🏅 Future Enhancements
- Expand disease classification to other plant species.
- Improve model performance with more data and advanced augmentation techniques.
- Enhance real-time detection through optimized webcam processing.

---

## 📚 Resources
- [TensorFlow Documentation](https://www.tensorflow.org/)  
- [LangChain Documentation](https://langchain.com/)  
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🏁 Final Thoughts
**Dr. Greenthumb** harnesses the power of machine learning and LLMs to bring a practical, user-friendly solution to plant care. Whether diagnosing diseases or offering expert advice, this project aims to make plant care more accessible and engaging for all users.
