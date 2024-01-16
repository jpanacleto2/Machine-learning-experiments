import streamlit as st
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

def ToxicityModel():
    # Load data and model
    df = pd.read_csv(os.path.join('../Demos/data/CommentToxicity', 'train.csv', 'train.csv'))
    MAX_FEATURES = 200000
    vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')
    vectorizer.adapt(df['comment_text'].values)
    model = tf.keras.models.load_model('../Demos/models/toxicity.h5')
    
    # Function to load the trained model and perform classification
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

    # Example section with comments
    st.header("Examples:")
    st.write("Here are some example comments that you can use to test the model:")

    example_comments = [
        "This is a great article!",
        "I love this product, it works really well.",
        "I hate you and everything you do.",
    ]

    for example_comment in example_comments:
        st.write('Example Comment: ', example_comment)

    # Text area to enter the comment
    comment_text = st.text_area('Enter your comment here:', '')

    # Check if the text has been entered
    if st.button('Analyze Toxicity'):
        # Check if the text is valid
        if comment_text:
            # Apply the model
            result = score_comment(comment_text)
            # Display the results
            st.text(result)
        else:
            st.warning('Please enter a comment before analyzing.')
