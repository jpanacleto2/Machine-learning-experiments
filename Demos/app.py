import streamlit as st
from ToxicityModel import ToxicityModel
from LeafDiseaseModel import LeafDiseaseModel

def Demos():
    st.title("Demos")
    st.write(
        "Welcome to our machine learning model demonstration application! "
        "Explore the demos below to experience and understand how our models work."
    )

    # Add more content as necessary

    st.subheader("Demo 1: Toxicity Detection in Comments")
    st.write(
        "In our first demo, we present a model trained to identify toxicity in comments. "
        "Enter a comment in the provided field and click 'Analyze Toxicity'. The model will provide probabilities "
        "of toxicity for various categories."
    )

    # Add examples of comments or links for users to try

    st.subheader("Demo 2: Plant Leaf Disease Classification")
    st.write(
        "Our second model is trained to classify images of plant leaves into three categories: "
        "Healthy, Powdery Mildew, and Rust. Upload an image of a leaf and see how the model predicts the disease category."
    )

    # Add examples of images or links for users to try

# Remaining code remains unchanged

# Sidebar configuration
st.sidebar.title("Choose Page")
selected_page = st.sidebar.radio("", ("Demos", "ToxicityModel", "LeafDiseaseModel"))

# Show the selected page
if selected_page == "Demos":
    Demos()
elif selected_page == "ToxicityModel":
    ToxicityModel()
elif selected_page == "LeafDiseaseModel":
    LeafDiseaseModel()
