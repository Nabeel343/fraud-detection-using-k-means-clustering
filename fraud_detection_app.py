# -----------------------------------------------------------
# Fraud Detection using KMeans + Streamlit Dashboard
# Raw Data -> Cleaning -> Training -> Visualization -> UI
# -----------------------------------------------------------

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



DATA_FILE = "fraud_dataset.csv"
CLEAN_FILE = "fraud_dataset_cleaned.csv"
MODEL_FILE = "fraud_kmeans_model.pkl"
SCALER_FILE = "scaler.pkl"



def generate_raw_dataset():
    np.random.seed(42)

    n = 300

    amount = np.concatenate([
        np.random.normal(500, 200, n),
        np.random.uniform(5000, 15000, 15)
    ])

    items = np.concatenate([
        np.random.randint(1, 6, n),
        np.random.randint(10, 25, 15)
    ])

    distance = np.concatenate([
        np.random.normal(3, 2, n),
        np.random.uniform(20, 60, 15)
    ])

    df = pd.DataFrame({
        "TransactionID": range(len(amount)),
        "Amount": amount,
        "Items": items,
        "Distance": distance
    })

    # missing values
    for col in df.columns[1:]:
        idx = np.random.choice(df.index, 10, replace=False)
        df.loc[idx, col] = np.nan

    df.to_csv(DATA_FILE, index=False)
    return df



def clean_data(df):
    df = df.copy()
    df = df.drop_duplicates()

    for col in ["Amount", "Items", "Distance"]:
        df[col] = df[col].fillna(df[col].median())

    df.to_csv(CLEAN_FILE, index=False)
    return df



def train_model(df):
    X = df[["Amount", "Items", "Distance"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    joblib.dump(kmeans, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    return kmeans, scaler, X_scaled



@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        kmeans = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    else:
        if not os.path.exists(DATA_FILE):
            df = generate_raw_dataset()
        else:
            df = pd.read_csv(DATA_FILE)

        df = clean_data(df)
        kmeans, scaler, _ = train_model(df)

    return kmeans, scaler



st.set_page_config(page_title="Fraud Detection AI", layout="wide")

st.title("🛡️ Fraud Detection System using K-Means Clustering")

kmeans, scaler = load_model()



df = pd.read_csv(CLEAN_FILE) if os.path.exists(CLEAN_FILE) else pd.read_csv(DATA_FILE)
df = df.dropna()


X = df[["Amount", "Items", "Distance"]]
X_scaled = scaler.transform(X)

clusters = kmeans.predict(X_scaled)
df["Cluster"] = clusters

distances = kmeans.transform(X_scaled)
df["RiskScore"] = np.min(distances, axis=1)



st.sidebar.header("Enter Transaction Details")

amount = st.sidebar.number_input("Amount", 0.0, 20000.0, 500.0)
items = st.sidebar.number_input("Items", 1, 30, 2)
distance = st.sidebar.number_input("Distance", 0.0, 100.0, 3.0)

new_data = np.array([[amount, items, distance]])
new_scaled = scaler.transform(new_data)

new_dist = kmeans.transform(new_scaled)
risk_score = np.min(new_dist)
pred_cluster = np.argmin(new_dist)

threshold = np.percentile(df["RiskScore"], 85)
risk = "🚨 Fraudulent" if risk_score > threshold else "✅ Normal"



st.subheader("Prediction Result")

st.write("Cluster:", int(pred_cluster))
st.write("Risk Score:", float(risk_score))

if risk == "🚨 Fraudulent":
    st.error(risk)
else:
    st.success(risk)



st.subheader("📊 Cluster Visualization (PCA)")

pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap="viridis", alpha=0.6)
ax.set_title("KMeans Clusters (PCA Reduced)")
st.pyplot(fig)



st.subheader("📈 Risk Score Distribution")

fig2, ax2 = plt.subplots()
ax2.hist(df["RiskScore"], bins=30)
ax2.axvline(threshold, color="red", linestyle="dashed")
ax2.set_title("Fraud Risk Score Distribution")
st.pyplot(fig2)



st.subheader("💳 Transaction Pattern")

fig3, ax3 = plt.subplots()
ax3.scatter(df["Amount"], df["Items"], c=clusters, alpha=0.5)
ax3.set_xlabel("Amount")
ax3.set_ylabel("Items")
ax3.set_title("Amount vs Items Clustering")
st.pyplot(fig3)



st.caption("Fraud Detection AI System | KMeans + Streamlit + PCA Visualization")