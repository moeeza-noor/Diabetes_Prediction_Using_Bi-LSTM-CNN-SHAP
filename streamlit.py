from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import shap
import pickle
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

# Load the pre-trained model
model = tf.keras.models.load_model('Hyperparameter_HybridBCNN_model.h5')  # Update with your model path

# Initialize the StandardScaler
scaler_mean = np.load('scaler_mean.npy')  # Update with your scaler mean path
scaler_scale = np.load('scaler_scale.npy')  # Update with your scaler scale path
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

# Define class labels
class_labels = {
    0: "No",
    1: "Yes",
    2: "No, borderline diabetes",
    3: "Yes (during pregnancy)"
}

# Feature details based on your dataset
features = {
    "HeartDisease": {"No": 0, "Yes": 1},
    "BMI": [12.02, 94.85],
    "Smoking": {"No": 0, "Yes": 1},
    "AlcoholDrinking": {"No": 0, "Yes": 1},
    "Stroke": {"No": 0, "Yes": 1},
    "PhysicalHealth": [0.0, 30.0],
    "MentalHealth": [0.0, 30.0],
    "DiffWalking": {"No": 0, "Yes": 1},
    "Sex": {"Female": 0, "Male": 1},
    "AgeCategory": {"18-24": 0, "80 or older": 1, "65-69": 2, "75-79": 3, "40-44": 4, "70-74": 5, "60-64": 6, "50-54": 7, "45-49": 8, "35-39": 9, "30-34": 10, "25-29": 11},
    "Race": {"American Indian/Alaskan Native": 0, "White": 1, "Black": 2, "Asian": 3, "Other": 4, "Hispanic": 5},
    "PhysicalActivity": {"No": 0, "Yes": 1},
    "GenHealth": {"Excellent": 0, "Very good": 1, "Fair": 2, "Good": 3, "Poor": 4},
    "SleepTime": [1.0, 24.0],
    "Asthma": {"No": 0, "Yes": 1},
    "KidneyDisease": {"No": 0, "Yes": 1},
    "SkinCancer": {"No": 0, "Yes": 1}
}

# Load SHAP values
with open('shap_values.pkl', 'rb') as f:
    shap_values = pickle.load(f)

# Define the Streamlit app
def main():
    # Custom CSS for styling
    st.markdown("""
        <style>
        .stApp {
            background-color: #eaf2f8;
        }
        .stSidebar {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title('🌟 Diabetes Prediction Model')
    st.markdown('### Enter the values for the following features to predict diabetes.')

    # Sidebar for input features
    st.sidebar.header('Input Features')
    st.sidebar.markdown('Fill in the details below:')

    input_data = {}
    for feature, value_range in features.items():
        if isinstance(value_range, dict):  # Handle categorical features
            selected_value = st.sidebar.radio(f"Select {feature}", list(value_range.keys()))
            input_data[feature] = value_range[selected_value]
        elif isinstance(value_range[0], str):  # Handle categorical ranges
            st.sidebar.error("Categorical ranges are not supported in this version")
        else:
            min_value, max_value = value_range
            selected_value = st.sidebar.number_input(f"Enter {feature}", min_value=min_value, max_value=max_value, step=0.01)
            input_data[feature] = selected_value

    # Predict button
    if st.sidebar.button('Predict'):
        # Convert input data to array
        input_array = np.array([input_data[feature] for feature in features]).reshape(1, -1)

        # Scale numerical features
        input_scaled = scaler.transform(input_array)

        # Reshape for model input
        input_scaled_reshaped = np.expand_dims(input_scaled, axis=2)

        # Get predictions
        predictions = model.predict(input_scaled_reshaped)[0]
        predicted_class = np.argmax(predictions)

        # Debugging: Print statements to inspect values
        print("Input Data:", input_data)
        print("Input Array Shape:", input_array.shape)
        print("Scaled Input Shape:", input_scaled.shape)
        print("Predictions:", predictions)

        # Prepare data for pie chart
        labels = [class_labels[i] for i in range(len(predictions))]
        sizes = predictions

        # Display prediction result
        st.success(f'Predicted Outcome: {class_labels[predicted_class]}')

        # Explanation for prediction result
        st.markdown(f"The pie chart below shows the probabilities for each class predicted by the model. The model predicted '{class_labels[predicted_class]}' as the most likely outcome.")

        # Display interactive pie chart with Plotly
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=.3)])
        fig_pie.update_traces(marker=dict(colors=px.colors.sequential.Blues))
        st.plotly_chart(fig_pie)

        # Display SHAP explanation
        st.subheader('🔍 Feature Importance')

        # Explanation for SHAP values
        st.markdown("The bar chart below shows the importance of each feature for the prediction. Higher bars indicate features that had a larger impact on the model's prediction.")

        # Ensure SHAP values are compatible
        if np.array(shap_values).shape[1] == input_scaled.shape[1]:
            # Select the SHAP values corresponding to the predicted class
            shap_values_class = shap_values[:, :, predicted_class]

            # Create a SHAP summary plot
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values_class, input_scaled, feature_names=list(features.keys()), plot_type='bar', show=False)
            st.pyplot(fig)
        else:
            st.error("The shape of the SHAP values does not match the input data.")
            st.write(f"Expected input shape: {input_scaled.shape}")
            st.write(f"SHAP values shape: {np.array(shap_values).shape}")

if __name__ == '__main__':
    main()
