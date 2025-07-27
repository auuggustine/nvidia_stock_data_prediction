import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

# Load the NVIDIA dataset
file_path = 'C:/Users/DELL/OneDrive/Documents/Dataset/Nvidia_stock_data.csv'
nvidia_data = pd.read_csv(file_path)

# Display first few rows
print(nvidia_data.head())

# Calculate the difference in closing price
difference = nvidia_data['Close'].diff()

# Binary classification: 1 if price increased, else 0
nvidia_data['Outcome'] = difference.apply(lambda x: 1 if x > 0 else 0)

# Set first row to 0
nvidia_data.loc[nvidia_data.index[0], 'Outcome'] = 0

# Features and Target
X = nvidia_data.drop(['Date', 'Outcome'], axis=1)
y = nvidia_data['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel="sigmoid")
model.fit(X_train, y_train)

# Predictions
y_predict = model.predict(X_test)

# Accuracy and classification report
print("\nAccuracy:", accuracy_score(y_test, y_predict))
print("\nClassification Report:\n", classification_report(y_test, y_predict))

# Predict a single example
example = X_test[:1]  # First sample as 2D array
predicted_outcome = model.predict(example)[0]

print("\nPrediction for first test example:")
if predicted_outcome == 1:
    print("Price Increased")
else:
    print("Price Decreased or Same")

#Streamlit UI
st.title("📈 NVIDIA Stock Price Movement Predictor")
st.write("Predict if the stock price will **increase or decrease** based on input values.")

# Sidebar user input
st.sidebar.header("Enter Stock Values:")
open_price = st.sidebar.number_input("Open Price", min_value=0.0, value=float(X['Open'].mean()))
high_price = st.sidebar.number_input("High Price", min_value=0.0, value=float(X['High'].mean()))
low_price = st.sidebar.number_input("Low Price", min_value=0.0, value=float(X['Low'].mean()))
close_price = st.sidebar.number_input("Close Price", min_value=0.0, value=float(X['Close'].mean()))
volume = st.sidebar.number_input("Volume", min_value=0.0, value=float(X['Volume'].mean()))

# Prepare input for prediction
user_input = np.array([[open_price, high_price, low_price, close_price, volume]])
user_input_scaled = scaler.transform(user_input)

if st.sidebar.button("Predict"):
    prediction = model.predict(user_input_scaled)[0]
    
    if prediction == 1:
        st.success("✅ Prediction: **Price will Increase**")
    else:
        st.error("❌ Prediction: **Price will Decrease or Stay the Same**")

# 3. Show Model Performance
st.subheader("Model Performance on Test Data:")
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"**Accuracy:** {accuracy:.2f}")
st.text(classification_report(y_test, model.predict(X_test)))