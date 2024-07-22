import streamlit as st
import numpy as np
import pandas as pd
import pickle


# Function to load the classifier model
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Load the trained model and encoders
classifier = load_model('classifier.pkl')
encoders = load_model('encoders.pkl')


def predict_note_authentication(features):
    # Define which columns are categorical
    categorical_columns = ["country", "location_type", "cellphone_access", "gender_of_respondent",
                           "relationship_with_head", "marital_status", "education_level", "job_type"]

    # Convert features to DataFrame for easier processing
    input_df = pd.DataFrame([features], columns=[
        "country", "year", "uniqueid", "location_type", "cellphone_access",
        "household_size", "age_of_respondent", "gender_of_respondent",
        "relationship_with_head", "marital_status", "education_level", "job_type"
    ])

    # Encode categorical features and handle unseen labels
    for column in categorical_columns:
        if column in encoders:
            input_df[column] = input_df[column].apply(lambda x: x if x in encoders[column].classes_ else "Unknown")
            input_df[column] = encoders[column].transform(input_df[column])

    # Convert to numpy array for prediction
    features_array = input_df.values
    prediction = classifier.predict(features_array)
    return prediction[0]


def main():
    st.title("ML Model Prediction")
    st.markdown("""
        <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>
        <p style='text-align: center;'>Enter the values for the features:</p>
        </div>
        """, unsafe_allow_html=True)

    # Input fields for features
    feature1 = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
    feature2 = st.number_input("Year", min_value=2000, max_value=2024, step=1)
    #feature3 = st.text_input("Uniqueid")
    feature3=1
    feature4 = st.selectbox("Location Type", ["Urban", "Rural"])
    feature5 = st.radio('Cellphone Access', ['Yes', 'No'])
    feature6 = st.number_input("Household Size", min_value=1, step=1)
    feature7 = st.number_input("Age of Respondent", min_value=18, max_value=100, step=1)
    feature8 = st.selectbox("Gender of Respondent", ["Male", "Female"])
    feature9 = st.selectbox("Relationship with Head", ["Head of Household", "Spouse", "Child", "Parent", "Other"])
    feature10 = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Widowed", "Other"])
    feature11 = st.selectbox("Education Level",
                             ["No formal education", "Primary education", "Secondary education", "Higher education",
                              "Other"])
    feature12 = st.selectbox("Job Type", ["Self employed", "Government Dependent", "Formally employed Private",
                                          "Informally employed", "Other"])

    # Check if all fields are filled
    if not all([feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10,
                feature11, feature12]):
        st.error("Please fill all the fields before submitting.")
        return

    # Validation button
    result = ""
    if st.button("Predict"):
        try:
            features = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9,
                        feature10, feature11, feature12]
            result = predict_note_authentication(features)
            if result is not None:
                st.success(f'The output is: {result}')
            else:
                st.error("Prediction could not be made due to unseen labels in the input.")
        except ValueError as ve:
            st.error(f"ValueError: {ve}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.button("About"):
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()