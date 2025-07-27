import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------
# Custom CSS for Beautiful UI
# ---------------------------
st.set_page_config(page_title="NVIDIA Stock Predictor", page_icon="📈", layout="wide")

st.markdown("""
<style>
    body {
        background-color: #0f1117;
        color: #d1d5db;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        color: #76b900;
        font-size: 50px;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        background: #1e293b;
        text-align: center;
        color: #fff;
        font-size: 24px;
    }
    .success {
        color: #76b900;
        font-weight: bold;
        font-size: 28px;
    }
    .error {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 28px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 1. Load and Train Model
# ---------------------------
file_path = 'C:/Users/DELL/OneDrive/Documents/Dataset/Nvidia_stock_data.csv'
nvidia_data = pd.read_csv(file_path)

# Calculate outcome (1 if price increases else 0)
difference = nvidia_data['Close'].diff()
nvidia_data['Outcome'] = difference.apply(lambda x: 1 if x > 0 else 0)
nvidia_data.loc[nvidia_data.index[0], 'Outcome'] = 0

# Features and Target
X = nvidia_data.drop(['Date', 'Outcome'], axis=1)
y = nvidia_data['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVC model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------------------
# 2. UI Section
# ---------------------------
st.markdown("<h1 class='main-title'>📈 NVIDIA Stock Movement Predictor</h1>", unsafe_allow_html=True)
st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Nvidia_logo.png", width=150)

st.markdown("### Enter Stock Details Below 👇")

# Use sliders for input (user-friendly)
col1, col2, col3 = st.columns(3)
with col1:
    open_price = st.number_input("Open Price", min_value=0.0, value=float(X['Open'].mean()))
with col2:
    high_price = st.number_input("High Price", min_value=0.0, value=float(X['High'].mean()))
with col3:
    low_price = st.number_input("Low Price", min_value=0.0, value=float(X['Low'].mean()))

col4, col5 = st.columns(2)
with col4:
    close_price = st.number_input("Close Price", min_value=0.0, value=float(X['Close'].mean()))
with col5:
    volume = st.number_input("Volume", min_value=0.0, value=float(X['Volume'].mean()))

# Prediction button
if st.button("🔍 Predict"):
    user_input = np.array([[open_price, high_price, low_price, close_price, volume]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    if prediction == 1:
        st.markdown("<p class='success'>✅ Prediction: Price Will Increase</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='error'>❌ Prediction: Price Will Decrease or Stay Same</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# 3. Show Accuracy
# ---------------------------
st.subheader("Model Accuracy:")
st.progress(int(accuracy * 100))
st.write(f"**Accuracy:** {accuracy:.2%}")

# ---------------------------
# 4. Optional: Show Chart
# ---------------------------
st.subheader("NVIDIA Stock Closing Price Trend")
st.line_chart(nvidia_data['Close'])
