import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image

# Load the tokenizer and model for humor detection
tokenizer = AutoTokenizer.from_pretrained("mohameddhiab/humor-no-humor")
model = AutoModelForSequenceClassification.from_pretrained("mohameddhiab/humor-no-humor")

# Set the title and background
st.set_page_config(page_title="Humor Detection App", layout="centered")
st.title("🤖 Humor Detection with Transformers")

# Create tabs in the desired order
tabs = st.tabs(["😂 Enter a Joke", "📖 About the Model", "🎓 Credits"])

# Enter a Joke tab
with tabs[0]:
    st.header("Test Your Jokes")
    st.subheader("Ever wanted to crack a joke but didn't know if your friends would laugh? Now you can find out if your joke is funny before!")
    
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
        It retains 97% of BERT's language understanding while being 60% faster and reducing the model size by 40%. 
        This model is particularly useful for tasks such as sentiment analysis, question answering, and text classification.
    """)
    st.write("This model has been fine-tuned on a joke/no-joke dataset to effectively detect humor in text. See more specifics about the model used at [Humor Detection Model](https://huggingface.co/mohameddhiab/humor-no-humor)")
    

# Credits tab
with tabs[2]:
    st.header("About Me")
    col1, col2 = st.columns([1, 2])  # Create two columns for layout
    
    with col1:
        # Display your permanent image
        image_path = "headshot.jpg"  # Replace with the name of your image file
        image = Image.open(image_path)
        st.image(image, caption='Morgan Baker', use_column_width=True)

    with col2:
        st.write("👋 Hello! I'm an undergraduate student studying Data Science* and Economics. I made this app for my coursework and because my jokes often flop.")
    
    st.markdown("---")  # Add a horizontal line for separation
    st.write("### Credits")
    st.write("App Developed by: Morgan Baker")
    st.write("Model used: [Humor Detection Model](https://huggingface.co/mohameddhiab/humor-no-humor)") 





