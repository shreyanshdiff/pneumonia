import streamlit as st
from keras.models import load_model
from PIL import Image
from sklearn.metrics import accuracy_score , precision_score ,recall_score , f1_score , confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from util import classify

def display_metrics(y_true, y_pred, class_names):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    confusion = confusion_matrix(y_true, y_pred)

    # Display metrics
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Display confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot()

st.title("PNEUMONIA CLASSIFIER")
st.header("Upload a X-RAY image of your chest")
file  = st.file_uploader("UPLOAD YOUR FILES HER" , type=['jpeg' , 'jpg' , 'png'])

model = load_model('pneumonia_classifier.h5')

with open('labels.txt' , 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
    
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image , use_column_width=True )

    class_name , conf_score = classify(image , model , class_names)

    st.write("## {}".format(class_name))
    st.write("### RATE : {}".format(int(conf_score * 1000) / 10))


    
    