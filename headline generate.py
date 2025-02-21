import streamlit as st
from transformers import pipeline

# Initialize the text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Set up Streamlit interface
st.title("Automated Car Parking Ad Headline Generator")
st.write("Enter a description or prompt for generating ad headlines for automated car parking.")

# Input prompt
prompt = st.text_input("Enter a prompt:", "Generate an ad headline for automated car parking:")

# Number of sequences to generate
num_return_sequences = st.slider("Number of headlines to generate", 1, 10, 5)

# Maximum length for the generated text
max_length = st.slider("Max length of generated text", 10, 50, 30)

# Generate text when the button is clicked
if st.button("Generate Headline"):
    with st.spinner("Generating headlines..."):
        try:
            # Generate output using the model
            output = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
            
            # Check if there is any output from the generator
            if not output:
                st.write("No headlines generated. Try adjusting the settings.")
            else:
                # Display the generated headlines
                for idx, text in enumerate(output):
                    st.subheader(f"Headline {idx+1}")
                    st.write(text["generated_text"])
        except Exception as e:
            st.error(f"An error occurred: {e}")
