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

# Credits section
st.header("Credits")
st.write("Model: `colesnic/distilbert-base-uncased-finetuned-ai-humor` by [colesnic](https://huggingface.co/colesnic/distilbert-base-uncased-finetuned-ai-humor)")

# Professional biography section
image = Image.open("headshot.jpg")  # Replace with your image file name

st.header("Morgan Baker")
st.image(image, caption='Morgan Baker', use_column_width=True)
st.write("Hello! I am an undergraduate student studying Data Science and Economics.")

# User guidance section
st.header("User Guidance")
st.write("Try puns or one-liners for the best results!")
st.write("Examples:")
st.write("- Input: 'Why did the chicken cross the road?'")
st.write("- Output: Humor")

# Conclusion section
st.header("Thank You!")
st.write("Thank you for using Humor Detector! We hope you found some laughs along the way.")
