# Streamlit Code
import streamlit as st
from inference import predict_species, load_model

# Load the model
model = load_model()

# Initialize Streamlit app
st.set_page_config(page_title="Iris Species Predictor")

# App title and description
st.title("Iris Species Predictor")
st.subheader("By Abhishek Nayar")
st.subheader("Predict the species of an iris flower based on its features")

# Input fields for user to enter flower features
sep_len = st.number_input("Sepal Length (cm)", min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width (cm)", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length (cm)", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width (cm)", min_value=0.00, step=0.01)

# Button to trigger prediction
button = st.button("Predict Species", type="primary")

if button:
    if sep_len and sep_wid and pet_len and pet_wid:
        # Perform prediction
        pred, probs_df = predict_species(model, sep_len, sep_wid, pet_len, pet_wid)
        
        # Display the prediction result
        st.success(f"Predicted Species: {pred}")
        
        # Display the prediction probabilities
        st.subheader("Prediction Probabilities")
        st.dataframe(probs_df)
    else:
        st.error("Please enter all the features to make a prediction.")

    st.bar_chart(probs_df.T)