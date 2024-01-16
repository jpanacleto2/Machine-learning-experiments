import streamlit as st
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn


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


    # Function to preprocess the image
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.Resize((225, 225)),
            transforms.ToTensor(),
        ])
        input_image = transform(image)
        input_image = input_image.unsqueeze(0)
        return input_image

    # Class mapping dictionary
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

    # Example section with images
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
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Input Image', use_column_width=True)

        # Preprocess the image
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
