import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("antibiotic_resistance_model.h5")
    return model

model = load_model()

# Title
st.title("Antibiotic Resistance Prediction from Gene Sequences")

# File uploader for gene sequence data
uploaded_file = st.file_uploader("Upload a CSV file containing gene sequences", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.write(data.head())
    
    # Ensure the correct column is used
    sequence_column = st.selectbox("Select the column containing gene sequences", data.columns)
    sequences = data[sequence_column].astype(str).tolist()
    
    # Tokenization and Padding (Assuming same preprocessing as training)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sequences)
    sequences = tokenizer.texts_to_sequences(sequences)
    sequences = pad_sequences(sequences, maxlen=200)
    
    # Predictions
    predictions = model.predict(sequences)
    predicted_classes = (predictions > 0.5).astype(int)
    data["Predicted_Resistance"] = predicted_classes
    
    # Display results
    st.write("Predictions:")
    st.write(data[[sequence_column, "Predicted_Resistance"]].head())
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    true_labels = st.file_uploader("Upload true labels CSV (if available)", type=["csv"])
    
    if true_labels is not None:
        true_data = pd.read_csv(true_labels)
        true_classes = true_data["Resistance"].values  # Assuming column name
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        st.pyplot(plt)
        
        # Classification Report
        report = classification_report(true_classes, predicted_classes, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

# Run: Save the model file as "antibiotic_resistance_model.h5" in the app directory.
