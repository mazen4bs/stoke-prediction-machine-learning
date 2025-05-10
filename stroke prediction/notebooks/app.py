import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🧠",
    layout="wide"
)

# Define base path for models
BASE_PATH = r"D:\semster7\ml_stroke\stoke-prediction-machine-learning\stroke prediction\notebooks"

# Define available models - add your models here
AVAILABLE_MODELS = {
    "SVM with SMOTE": os.path.join(BASE_PATH, r"D:\semster7\ml_stroke\stoke-prediction-machine-learning\stroke prediction\notebooks\svm_model_smote2.pkl"),
    "SVM without SMOTE": os.path.join(BASE_PATH, r"D:\semster7\ml_stroke\stoke-prediction-machine-learning\stroke prediction\notebooks\svm_model.pkl"),
     "KNN without SMOTE": os.path.join(BASE_PATH, r"D:\semster7\ml_stroke\stoke-prediction-machine-learning\stroke prediction\notebooks\Knn_model.pkl"),
     "knn with SMOTE": os.path.join(BASE_PATH, r"D:\semster7\ml_stroke\stoke-prediction-machine-learning\stroke prediction\notebooks\knn_model_smote.pkl"),
     "svm  tuned": os.path.join(BASE_PATH, r"D:\semster7\ml_stroke\stoke-prediction-machine-learning\stroke prediction\notebooks\svm_model_tuned.pkl"),
    # Add more models here as needed
    # "Random Forest": os.path.join(BASE_PATH, "rf_model.pkl"),
    # "Logistic Regression": os.path.join(BASE_PATH, "lr_model.pkl"),
}

# Path for the scaler
SCALER_PATH = os.path.join(BASE_PATH, "scaler.pkl")

# Create a function to load models with error handling
@st.cache_resource
def load_models_and_scaler():
    models = {}
    scaler = None
    
    # Load scaler
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        st.error(f"Error loading scaler: {e}")
        st.info("Please check if the scaler file exists at the specified path.")
    
    # Load models
    for model_name, model_path in AVAILABLE_MODELS.items():
        try:
            models[model_name] = joblib.load(model_path)
        except FileNotFoundError as e:
            st.error(f"Error loading {model_name} model: {e}")
            st.info(f"Please check if the model file exists at: {model_path}")
    
    return models, scaler

# Load models and scaler
models, scaler = load_models_and_scaler()

# Title and description
st.title("🧠 Stroke Prediction App")
st.markdown("""
This application uses machine learning models to predict stroke risk based on your health information.
Fill in the form below to get a personalized risk assessment.
""")

# Model selection
model_name = st.sidebar.selectbox(
    "Select Prediction Model",
    list(AVAILABLE_MODELS.keys()),
    index=0,
    help="Choose which model to use for prediction"
)

# Show model information
with st.sidebar.expander(f"About {model_name}"):
    if model_name == "SVM with SMOTE":
        st.markdown("""
        **SVM with SMOTE**
        
        This model uses Support Vector Machine and was trained with Synthetic Minority 
        Over-sampling Technique (SMOTE) to address class imbalance.
        
        - Handles imbalanced data better
        - May be more sensitive to minority class (stroke cases)
        """)
    elif model_name == "SVM without SMOTE":
        st.markdown("""
        **SVM without SMOTE**
        
        This is a standard Support Vector Machine model trained on the original dataset 
        without any oversampling or undersampling techniques.
        
        - May be more conservative in predictions
        - Optimized for overall accuracy
        """)
    # Add descriptions for other models here

# Create columns for better layout
col1, col2 = st.columns(2)

# User Inputs
with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked", "children"])

with col2:
    st.subheader("Health Information")
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", min_value=40.0, max_value=400.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Process input to match training data preprocessing
def preprocess_input(age_val, glucose_val, bmi_val, hypertension_val, heart_disease_val, 
                     gender_val, married_val, work_type_val, residence_val, smoking_val):
    # Create input dictionary with explicitly passed parameters to avoid scope issues
    input_dict = {
        "age": age_val,
        "avg_glucose_level": glucose_val,
        "bmi": bmi_val,
        "hypertension": hypertension_val,
        "heart_disease": heart_disease_val,
        "gender": gender_val,
        "ever_married": married_val,
        "work_type": work_type_val,
        "Residence_type": residence_val,
        "smoking_status": smoking_val
    }
    
    # Create dataframe
    input_df = pd.DataFrame([input_dict])
    
    # Map categorical values exactly as they were during training
    # Creating dummy variables manually to ensure consistency with training
    
    # Gender encoding
    input_df['gender_Male'] = input_df['gender'] == 'Male'
    input_df['gender_Other'] = input_df['gender'] == 'Other'
    
    # Ever married encoding
    input_df['ever_married_Yes'] = input_df['ever_married'] == 'Yes'
    
    # Work type encoding (with Non-working grouping)
    input_df['work_type_Non-working'] = input_df['work_type'].isin(['Never_worked', 'children'])
    input_df['work_type_Private'] = input_df['work_type'] == 'Private'
    input_df['work_type_Self-employed'] = input_df['work_type'] == 'Self-employed'
    
    # Residence type encoding
    input_df['Residence_type_Urban'] = input_df['Residence_type'] == 'Urban'
    
    # Smoking status encoding (dropping 'formerly smoked' as reference)
    input_df['smoking_status_never smoked'] = input_df['smoking_status'] == 'never smoked'
    input_df['smoking_status_smokes'] = input_df['smoking_status'] == 'smokes'
    
    # Drop original categorical columns
    input_df = input_df.drop(columns=categorical_cols)
    
    # Define all expected columns in training - exactly matching the model's training data
    expected_cols = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
        'gender_Male', 'gender_Other',
        'ever_married_Yes',
        'work_type_Non-working', 'work_type_Private', 'work_type_Self-employed',
        'Residence_type_Urban',
        'smoking_status_never smoked', 'smoking_status_smokes'
    ]
    
    # Handle column order and missing columns
    # Create a DataFrame with all zeros (or False) for expected columns
    empty_df = pd.DataFrame({col: [False if col.startswith('gender_') or 
                                    col.startswith('ever_married_') or 
                                    col.startswith('work_type_') or 
                                    col.startswith('Residence_type_') or 
                                    col.startswith('smoking_status_') else 0] 
                               for col in expected_cols})
    
    # Update with actual values from input_df where they exist
    for col in input_df.columns:
        if col in expected_cols:
            empty_df[col] = input_df[col]
    
    # Replace input_df with properly formatted DataFrame
    input_df = empty_df[expected_cols]
    
    # Scale numeric features
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    if scaler is not None:
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    return input_df

# Add a predict button
if st.button("Predict Stroke Risk"):
    # Check if we have the selected model loaded
    if model_name not in models:
        st.error(f"The selected model '{model_name}' is not available. Please check the model files.")
    elif scaler is None:
        st.error("Scaler not loaded. Please check the scaler file path.")
    else:
        try:
            # For debugging
            st.write(f"Using model: {model_name}")
            
            # Get the selected model
            selected_model = models[model_name]
            
            # Define categorical columns for preprocessing
            categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            
            # Preprocess input with explicitly passed parameters
            input_df = preprocess_input(
                age_val=age,
                glucose_val=avg_glucose_level,
                bmi_val=bmi,
                hypertension_val=hypertension,
                heart_disease_val=heart_disease,
                gender_val=gender,
                married_val=ever_married,
                work_type_val=work_type,
                residence_val=residence_type,
                smoking_val=smoking_status
            )
            
            # Debug: Show processed input
            with st.expander("View preprocessed data"):
                st.write(input_df)
            
            # Make prediction with selected model
            prediction_result = selected_model.predict(input_df)[0]
            probability_result = selected_model.predict_proba(input_df)[0][1]
            
            # Display results
            st.header("Prediction Result")
            
            # Create columns for risk level display
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if prediction_result == 1:
                    st.error("⚠️ **High Stroke Risk Detected**")
                else:
                    st.success("✅ **Low Stroke Risk Detected**")
                
                st.metric("Risk Probability", f"{probability_result:.2%}")
            
            with res_col2:
                # Risk interpretation
                if probability_result < 0.2:
                    risk_level = "Low"
                    recommendation = "Maintain a healthy lifestyle with regular exercise and balanced diet."
                elif probability_result < 0.5:
                    risk_level = "Moderate"
                    recommendation = "Consider discussing your risk factors with a healthcare provider."
                else:
                    risk_level = "High"
                    recommendation = "Please consult with a healthcare professional as soon as possible."
                
                st.info(f"**Risk Level**: {risk_level}")
                st.markdown(f"**Recommendation**: {recommendation}")
                st.warning("**Disclaimer**: This is a machine learning prediction and should not replace professional medical advice.")
            
            # Add a note about the selected model
            st.caption(f"Prediction made using the {model_name} model")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.info("Please ensure all fields are filled correctly.")
            st.write("Please try again or select a different model.")

# Add information about the app
with st.expander("About this App"):
    st.markdown("""
    ## Stroke Prediction App
    
    This application uses machine learning models to assess stroke risk based on various health and demographic factors.
    
    ### Features used by the models:
    - Age
    - Gender
    - Hypertension
    - Heart Disease
    - Ever Married
    - Work Type
    - Residence Type
    - Average Glucose Level
    - BMI
    - Smoking Status
    
    ### Available Models:
    - **SVM with SMOTE**: Support Vector Machine trained with synthetic minority oversampling to handle class imbalance
    - **SVM without SMOTE**: Standard Support Vector Machine trained on the original dataset
    
    **Note**: This model is for educational purposes only and should not replace professional medical advice.
    """)