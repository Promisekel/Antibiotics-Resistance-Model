import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from Bio import SeqIO
import io

# App title
st.title("Antibiotic Resistance Prediction")
st.markdown("## Predict antibiotic resistance from genomic sequences")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset (.fasta, .fastq, or .npy file)", type=["fasta", "fastq", "npy"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".npy"):
        DataRaw = np.load(uploaded_file, allow_pickle=True)
        Datadict = DataRaw[()]
        DataDf = pd.DataFrame.from_dict(Datadict)
    else:
        file_content = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        sequences = []
        labels = []
        file_format = "fastq" if uploaded_file.name.endswith(".fastq") else "fasta"
        
        for record in SeqIO.parse(file_content, file_format):
            sequences.append(str(record.seq))
            labels.append(1 if "resistant" in record.description.lower() else 0)
        
        DataDf = pd.DataFrame({"genes": sequences, "resistant": labels})
    
    st.write("### Preview of Uploaded Data:")
    st.dataframe(DataDf.head())

    # Tokenize genomic sequences
    maxlen = 160
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(list(DataDf['genes']))
    sequences = tokenizer.texts_to_sequences(list(DataDf['genes']))
    Xpad = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post', value=0)
    labels = np.asarray(DataDf['resistant'])

    # Train-Test Split
    training_samples = int(Xpad.shape[0] * 0.9)
    indices = np.arange(Xpad.shape[0])
    np.random.shuffle(indices)
    Xpad = Xpad[indices]
    labels = labels[indices]
    x_train, y_train = Xpad[:training_samples], labels[:training_samples]
    x_test, y_test = Xpad[training_samples:], labels[training_samples:]

    # Model Architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 1, input_length=maxlen),
        tf.keras.layers.Conv1D(128, 27, activation='relu'),
        tf.keras.layers.MaxPooling1D(9),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv1D(128, 9, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.2)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st.write("### Model Summary:")
    st.text(model.summary())

    # Train model
    if st.button("Train Model"):
        history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
        st.success("Model training completed!")

        # Plot learning curves
        st.write("### Training & Validation Loss")
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

        st.write("### Training & Validation Accuracy")
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Training Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)

    # Make Predictions
    if st.button("Predict on Test Data"):
        predictions = model.predict(x_test)
        preds_binary = (predictions >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, preds_binary)
        precision = precision_score(y_test, preds_binary)
        recall = recall_score(y_test, preds_binary)
        f1 = f1_score(y_test, preds_binary)
        st.write(f"### Model Performance on Test Data")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, preds_binary)
        st.write("### Confusion Matrix")
        st.write(conf_matrix)

st.markdown("### Hospital Usage Features:")
st.markdown("- Upload patient genomic data for resistance prediction")
st.markdown("- View model accuracy and performance metrics")
st.markdown("- Train and retrain model with updated datasets")
