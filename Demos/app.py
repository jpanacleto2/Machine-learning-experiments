import streamlit as st
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn

def Demos():
    st.title("Demos")
    st.write(
        "Welcome to our machine learning model demonstration application! "
        "Explore the demos below to experience and understand how our models work."
    )


    st.subheader("Demo 1: Toxicity Detection in Comments")
    st.write(
        "In our first demo, we present a model trained to identify toxicity in comments. "
        "Enter a comment in the provided field and click 'Analyze Toxicity'. The model will provide probabilities "
        "of toxicity for various categories."
    )


    st.subheader("Demo 2: Plant Leaf Disease Classification")
    st.write(
        "Our second model is trained to classify images of plant leaves into three categories: "
        "Healthy, Powdery Mildew, and Rust. Upload an image of a leaf and see how the model predicts the disease category."
    )

def ToxicityModel():
    # Load data and model
    df = pd.read_csv(os.path.join('../Demos/data/CommentToxicity', 'train.csv', 'train.csv'))
    MAX_FEATURES = 200000
    vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
    vectorizer.adapt(df['comment_text'].values)
    model = tf.keras.models.load_model('../Demos/models/toxicity.h5')
    
    # load the trained model and perform classification
    def score_comment(comment):
        vectorized_comment = vectorizer([comment])
        results = model.predict(vectorized_comment)

        text = ''
        for idx, col in enumerate(df.columns[2:]):
            text += '{}: {}\n'.format(col, results[0][idx] > 0.5)

        return text

    st.title('Toxicity Detection in Comments')

    st.write(
        "This is a deep learning model trained to identify toxicity in comments. Enter a comment in the field below "
        "and click the button to analyze the presence of toxicity. The model will provide toxicity probabilities for various categories."
    )

    
    st.header("Examples:")
    st.write("Here are some example comments that you can use to test the model:")

    example_comments = [
        "This is a great article!",
        "I love this product, it works really well.",
        "I hate you and everything you do.",
    ]

    for example_comment in example_comments:
        st.write('Example Comment: ', example_comment)

    # Text area
    comment_text = st.text_area('Enter your comment here:', '')

    # Check if the text has been entered
    if st.button('Analyze Toxicity'):
        # Check if the text is valid
        if comment_text:
            # Apply the model
            result = score_comment(comment_text)
            
            st.text(result)
        else:
            st.warning('Please enter a comment before analyzing.')

def LeafDiseaseModel():
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * 56 * 56, 64)  # Adjust the input size based on your image size
            self.fc2 = nn.Linear(64, 3)  # Adjust the output size based on your number of classes

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(x)
            x = F.relu(self.conv2(x))
            x = self.pool2(x)
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x  # No need to apply softmax here for Streamlit


    # preprocess the image
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.Resize((225, 225)),
            transforms.ToTensor(),
        ])
        input_image = transform(image)
        input_image = input_image.unsqueeze(0)
        return input_image

    # mapping dictionary
    class_mapping = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

    # Load PyTorch model
    model = SimpleCNN()
    model.load_state_dict(torch.load('../Demos/models/model.pth'))
    model.eval()

    st.title("Plant Leaf Disease Classification Model")

    st.write(
        "This is a deep learning model trained to classify images of plant leaves into three categories: Healthy, "
        "Powdery and Rust. Upload an image of a leaf and see how the model predicts the disease category."
    )

    
    st.header("Examples:")
    st.write("Here are some example images that you can use to test the model:")

    example_images = [
        "../Demos/assets/8bc2979962db6549.jpg",
        "../Demos/assets/8a2d598f2ec436e6.jpg",
        "../Demos/assets/8a954b82bf81f2bc.jpg",
    ]

    for example_image_url in example_images:
        st.image(example_image_url, caption='Example', use_column_width=True)

    uploaded_file = st.file_uploader("Or upload your own image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption='Input Image', use_column_width=True)

        
        input_image = preprocess_image(image)

        # Make the prediction
        with torch.no_grad():
            output = model(input_image)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(output[0], dim=0)

        # Get the predicted class
        predicted_class = torch.argmax(probabilities).item()
        predicted_class_label = class_mapping[predicted_class]

        st.write(f"The predicted class is: {predicted_class_label}")

st.sidebar.title("Choose Page")
selected_page = st.sidebar.radio("", ("Demos", "ToxicityModel", "LeafDiseaseModel"))


if selected_page == "Demos":
    Demos()
elif selected_page == "ToxicityModel":
    ToxicityModel()
elif selected_page == "LeafDiseaseModel":
    LeafDiseaseModel()
