# app.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the breast cancer dataset
df = pd.read_csv('data_BreastCancer.csv')
df = df.drop(['Unnamed: 32'], axis=1)

# Select features and labels
features = df.drop(['id', 'diagnosis'], axis=1)
labels = df['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Impute and scale data
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Train the KNN model
k_value = 3
knn_model = KNeighborsClassifier(n_neighbors=k_value)
knn_model.fit(X_train_scaled, y_train)

# Streamlit app
st.title("Breast Cancer Diagnosis Prediction")

# Sidebar for user input
st.sidebar.header("Dữ liệu đầu vào người dùng:")
user_input = {}

for feature in features.columns:
    user_input[feature] = st.sidebar.slider(f"Select {feature}", float(features[feature].min()), float(features[feature].max()), float(features[feature].mean()))

# Prepare input data for prediction
new_data = pd.DataFrame(user_input, index=[0])
new_data_imputed = imputer.transform(new_data)
new_data_scaled = scaler.transform(new_data_imputed)

# Make prediction
prediction = knn_model.predict(new_data_scaled)

# Display prediction
st.subheader("Kết quả dự đoán:")
st.write(f"Kết quả: {prediction[0]}")

# Display the dataset and model accuracy
st.subheader("Độ chính xác của dữ liệu và mô hình")
st.write(f"Hình dạng tập dữ liệu gốc: {df.shape}")
st.write(f"Độ chính xác của mô hình trên dữ liệu thử nghiệm: {accuracy_score(y_test, knn_model.predict(X_test_scaled)):.2f}")

