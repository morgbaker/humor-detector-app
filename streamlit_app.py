import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image

import streamlit as st
from PIL import Image  # If you're using images elsewhere
import os  # For handling file paths

st.set_page_config(page_title="Humor Detection App", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #c74848 !important;  /* Background color */
        background-image: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 0h20v20H0V0zm10 17a7 7 0 1 0 0-14 7 7 0 0 0 0 14zm20 0a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM10 37a7 7 0 1 0 0-14 7 7 0 0 0 0 14zm10-17h20v20H20V20zm10 17a7 7 0 1 0 0-14 7 7 0 0 0 0 14z' fill='%23ed8d50' fill-opacity='0.59' fill-rule='evenodd'/%3E%3C/svg%3E") !important; /* Background image */
        background-size: cover !important;
        background-repeat: no-repeat !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Load the tokenizer and model for humor detection
tokenizer = AutoTokenizer.from_pretrained("mohameddhiab/humor-no-humor")
model = AutoModelForSequenceClassification.from_pretrained("mohameddhiab/humor-no-humor")

# Set the title and background

st.title("🤖 Humor Detection with Transformers")

# Create tabs
tabs = st.tabs(["😂 Enter a Joke", "📖 About the Model", "🎓 Credits"])

# Enter a Joke tab
with tabs[0]:
    st.header("Test Your Jokes")
    st.subheader("Ever wondered if your joke would land? Now you can test your ideas and discover if your joke is funny before you share!")
    with st.expander("Disclaimer", expanded=False):
        st.write("Please note that the humor detection model may not always provide accurate results. The creators are not responsible for any embarrassment caused by jokes that may not land as intended.")
    # Text input from user
    input_text = st.text_input("💬 Enter your joke:", "")

    if input_text:
        # Tokenize the input text
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted class
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

        # Define the labels
        labels = ['No Humor', 'Humor']  # Adjust labels based on the model's training
        result = labels[predicted_class]

        # Display the result
        st.write("### Prediction:")
        st.write(result)

        # Fun effects based on the prediction
        if result == 'Humor':
            st.balloons()  # Display balloons effect
            st.success("😂 That's a funny joke! Keep them coming!")
        else:
            st.warning("😐 Not quite a joke! Better luck next time!")


# About the Model tab
with tabs[1]:
    st.header("About the Model")

    st.write("### DistilBERT")
    st.write("""
        DistilBERT is a smaller, faster, and lighter version of BERT (Bidirectional Encoder Representations from Transformers). 
        Developed by Hugging Face, it retains **97% of BERT's language understanding** while being **60% faster** and reducing the model size by **40%**. 
        This efficiency makes DistilBERT particularly suitable for applications where computational resources are limited or where quick responses are essential, such as:
    """)
    
    st.write("""
        - **Sentiment Analysis**: Understanding the sentiment conveyed in text data.
        - **Question Answering**: Providing precise answers to user queries.
        - **Text Classification**: Categorizing text into predefined labels.
    """)

    st.write("### Humor Detection")
    st.write("""
        This model has been **fine-tuned on a joke/no-joke dataset** to effectively detect humor in text. By leveraging the power of DistilBERT, the humor detection model can analyze linguistic patterns, context, and sentiment to discern whether a given piece of text is humorous or not.
        
        For more details about the specific model used, check out the [Humor Detection Model](https://huggingface.co/mohameddhiab/humor-no-humor).
    """)

    st.write("### Limitations and Things to Keep in Mind")
    st.write("""
        While DistilBERT is powerful, there are several limitations to consider when using this model for humor classification:
        
        - **Contextual Understanding**: Humor is often context-dependent, and the model may struggle to capture nuanced cultural references or context-specific humor.
        
        - **Subjectivity**: Humor is inherently subjective; what one person finds funny, another may not. The model's training data may not cover all variations of humor, leading to potential misclassifications.
        
        - **Limited Dataset**: The effectiveness of the model depends on the quality and diversity of the dataset it was fine-tuned on. This means the model may not generalize well to new or different types of humor.
        
        - **Language Nuances**: Sarcasm, irony, and wordplay can be particularly challenging for AI models to interpret correctly. The model may not always accurately identify these forms of humor.
    """)

    st.markdown("---")  # Add a horizontal line for separation


# Credits tab
with tabs[2]:
    st.header("About Me")
    col1, col2 = st.columns([1, 2])  
    
    with col1:
        image_path = "headshot.jpg"  
        image = Image.open(image_path)
        st.image(image, caption='Morgan Baker', use_column_width=True)

    with col2:
        st.write("👋 Hello! I'm an undergraduate student studying Data Science and Economics at West Virginia University. I am interested in humor text analysis and machine learning applications. Please feel free to reach out with any questions or feedback!")
        st.write("🔗 LinkedIn: [Connect with me!](https://www.linkedin.com/in/morgan-baker-1a358b265/)")
    st.markdown("---")  
    st.write("### Credits")
    st.write("App Developed by: Morgan Baker")
    st.write("Model used: [Humor Detection Model](https://huggingface.co/mohameddhiab/humor-no-humor)") 





