# 💳 Fraud Detection using K-Means Clustering

## 📌 Overview
This project detects potentially fraudulent transactions using **unsupervised machine learning (K-Means Clustering)**. Since fraud labels are often unavailable in real-world scenarios, clustering helps identify anomalous patterns based on transaction behavior.

---

## 🚀 Features
- Data preprocessing & feature scaling
- K-Means clustering for anomaly detection
- Fraud risk scoring system
- PCA-based visualization of clusters
- Interactive Streamlit web application

---

## 🧠 Problem Statement
Financial fraud causes significant losses globally. The goal of this project is to identify unusual transaction patterns without using labeled data, leveraging clustering techniques to detect anomalies.

---

## 🛠️ Tech Stack
- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Streamlit

---

## ⚙️ Workflow
1. Data Collection / Generation  
2. Data Cleaning & Preprocessing  
3. Feature Scaling  
4. K-Means Clustering  
5. Anomaly Detection (Distance-based scoring)  
6. Visualization using PCA  
7. Streamlit Dashboard Deployment  

---

## 📊 Results
- Successfully grouped transactions into normal and suspicious clusters  
- Identified high-risk transactions using centroid distance  
- Visualized clusters clearly using PCA  

---

## 🖥️ Run Locally

```bash
git clone https://github.com/your-username/fraud-detection-kmeans.git
cd fraud-detection-kmeans
pip install -r requirements.txt
streamlit run app.py
