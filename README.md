# Diabetes Prediction using Hybrid Deep Learning Bi-LSTM-CNN Model

**SugarCare** is a Streamlit application that leverages a pre-trained machine learning model to predict the likelihood of diabetes based on user input. The app provides a user-friendly interface for entering patient data, predicting outcomes, and visualizing results with interactive charts and SHAP (SHapley Additive exPlanations) values to explain model predictions.

## Features

- **User Input Interface:**
  - Enter values for various health-related features via a sidebar.
  - Input includes both categorical and numerical data.

- **Diabetes Prediction:**
  - Uses a pre-trained neural network model to predict diabetes risk based on input features.
  - Displays prediction results along with probabilities for each class.

- **Interactive Visualization:**
  - Pie chart visualizing the model's prediction probabilities for each outcome.
  - SHAP values bar chart showing the impact of each feature on the prediction.

## Model Architecture and Training

The model used in this application is a hybrid of Bidirectional LSTM (Bi-LSTM) and Convolutional Neural Network (CNN). Here is an overview of the model architecture and training process:

### Model Architecture

- **Bidirectional LSTM Branch:**
  - A Bi-LSTM model is used to capture temporal dependencies in sequential data.
  - The output of the last LSTM layer is used for concatenation with the CNN branch.

- **CNN Branch:**
  - Convolutional layers are used to extract local features from the input data.
  - The CNN branch consists of:
    - A Conv1D layer with 128 filters and a kernel size of 3, followed by a ReLU activation function.
    - A MaxPooling1D layer with a pool size of 2.
    - A Flatten layer to convert the 2D feature maps into a 1D vector.

- **Hybrid Model:**
  - The outputs of the Bi-LSTM and CNN branches are concatenated.
  - The concatenated output is fed into a Dense layer with a softmax activation function to produce the final predictions.

## Files Used

- **Hyperparameter_HybridBCNN_model.h5**: A pre-trained TensorFlow Keras model for predicting diabetes. This model is used to make predictions based on user input.

- **scaler_mean.npy** and **scaler_scale.npy**: NumPy files containing the mean and scale used for feature scaling. These files are used to standardize the input features before making predictions.

- **shap_values.pkl**: A Pickle file containing SHAP values for feature importance analysis. It helps explain the model's predictions by showing the contribution of each feature.

## Installation

To set up and run the Virtual Doctor application, follow these steps:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/virtual-doctor.git
    ```

2. **Navigate to the Project Directory:**

    ```bash
    cd virtual-doctor
    ```

3. **Install Dependencies:** Make sure you have Python installed, then install the required packages:

    ```bash
    pip install streamlit numpy tensorflow scikit-learn shap plotly matplotlib pandas
    ```

4. **Run the Application:**

    ```bash
    streamlit run app.py
    ```

    Replace `app.py` with the name of your Python file if different.

## Features and Techniques

- **Model Loading**: Loads a pre-trained TensorFlow Keras model to make predictions.
  
- **Feature Scaling**: Uses `StandardScaler` for scaling numerical features based on precomputed mean and scale.

- **SHAP Values**: Utilizes SHAP values for explaining model predictions and feature importance.

- **Interactive Charts**: Displays prediction probabilities and feature importance using Plotly and Matplotlib.

## Usage

1. **Input Data:**

    - Enter values for features in the sidebar. Adjust numerical values using sliders or input fields.

2. **Click "Predict"** to get the model's prediction.

3. **View Results:**

    - The predicted outcome will be displayed along with a pie chart showing prediction probabilities.
    - SHAP values will be displayed as a bar chart to explain the importance of each feature.

## Contributing

Feel free to submit issues or pull requests if you have improvements or fixes!
