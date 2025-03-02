import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# Load the trained Random Forest model
MODEL_PATH = "random_forest_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define feature names and class labels
FEATURES = ["temp", "humidity", "windspeed", "precipcover", "cloudcover", "sealevelpressure", "visibility"]
CLASS_LABELS = {0: "No Rain", 1: "Rain Likely"}

# Streamlit UI
st.title("Weather Prediction App ☁️🌧️")
st.write("Predict the likelihood of rain based on weather parameters.")

# Sidebar for user input
st.sidebar.header("Input Weather Metrics")
def get_user_input():
    """Get user inputs for weather features."""
    return pd.DataFrame([{
        "temp": st.sidebar.number_input("Temperature (°C)", value=25.0, step=0.1, format="%.1f"),
        "humidity": st.sidebar.number_input("Humidity (%)", value=60.0, step=0.1, format="%.1f"),
        "windspeed": st.sidebar.number_input("Wind Speed (km/h)", value=10.0, step=0.1, format="%.1f"),
        "precipcover": st.sidebar.number_input("Precipitation Cover (%)", value=30.0, step=0.1, format="%.1f"),
        "cloudcover": st.sidebar.number_input("Cloud Cover (%)", value=50.0, step=0.1, format="%.1f"),
        "sealevelpressure": st.sidebar.number_input("Sea Level Pressure (hPa)", value=1013.0, step=0.1, format="%.1f"),
        "visibility": st.sidebar.number_input("Visibility (km)", value=10.0, step=0.1, format="%.1f")
    }])

# Collect user input
data = get_user_input()

# Display user input
st.subheader("Your Input Data")
st.dataframe(data)

# Perform prediction
try:
    prediction = model.predict(data[FEATURES])[0]
    prediction_proba = model.predict_proba(data[FEATURES])[0] * 100
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

# Display prediction result
st.subheader("Prediction Result")
st.write(f"Predicted Status: **{CLASS_LABELS[prediction]}**")

# Display prediction probabilities
st.subheader("Prediction Probabilities")
probability_df = pd.DataFrame([prediction_proba], columns=CLASS_LABELS.values()).style.format("{:.2f}%")
st.dataframe(probability_df)

# Generate pie chart for prediction probabilities
fig, ax = plt.subplots()
ax.pie(prediction_proba, labels=CLASS_LABELS.values(), autopct="%.1f%%", colors=["#ff9999", "#66b3ff"], startangle=90)
ax.set_title("Prediction Probabilities")

# Display pie chart in Streamlit
buf = BytesIO()
plt.savefig(buf, format="png")
st.image(buf)
buf.close()

# Additional Insights
st.subheader("Weather Condition Assessment")
if prediction == 1:
    st.warning("High chance of rain. Consider carrying an umbrella or raincoat.")
else:
    st.success("No rain expected. Enjoy the weather!")

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit.")
