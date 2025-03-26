

import os
import streamlit as st
import google.generativeai as genai
from google.genai import types
from dotenv import load_dotenv
import time


# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Levrage AI — Image Generation", layout="centered")
st.title("Levrage AI — Prompt-to-Image Generator")

# Session state to preserve last generated image
if "last_generated_image" not in st.session_state:
    st.session_state.last_generated_image = None

# Initialize Google GenAI client
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

# Use supported or fallback model
MODEL =  "gemini-2.0-flash-exp-image-generation"

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    return file_name


def generate_image(prompt, reference_image_path=None):
    parts = []

    if reference_image_path and os.path.exists(reference_image_path):
        uploaded = client.files.upload(file=reference_image_path)
        parts.append(
            types.Part.from_uri(
                file_uri=uploaded.uri,
                mime_type=uploaded.mime_type,
            )
        )

    parts.append(types.Part(text=prompt))
    contents = [types.Content(role="user", parts=parts)]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_modalities=["image", "text"],
        response_mime_type="text/plain"
    )

    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data:
                    file_path = f"generated_image_{int(time.time())}.png"
                    save_binary_file(file_path, part.inline_data.data)
                    return file_path
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# --- UI Layout ---

st.subheader("Input Prompt")
user_prompt = st.text_area(
    label="Describe the image you want to generate",
    placeholder="Example: A futuristic city skyline at dusk with flying cars",
    height=150
)

st.subheader("Reference Image (Optional)")
uploaded_image = st.file_uploader("Upload an image to guide the generation", type=["png", "jpg", "jpeg"])

use_last = st.checkbox("Use previously generated image as reference")

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    generate = st.button("Generate Image")

if generate:
    if not user_prompt.strip():
        st.warning("Please enter a valid prompt.")
    else:
        st.info("Generating image, please wait...")
        reference_path = None

        if uploaded_image:
            temp_path = "uploaded_reference.png"
            with open(temp_path, "wb") as f:
                f.write(uploaded_image.read())
            reference_path = temp_path
        elif use_last and st.session_state.last_generated_image:
            reference_path = st.session_state.last_generated_image

        image_path = generate_image(user_prompt, reference_image_path=reference_path)

        if image_path:
            st.session_state.last_generated_image = image_path
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()
                st.success("Image generated successfully.")
                st.image(image_bytes, caption="Generated Image", use_container_width=True)
                st.download_button("Download Image", data=image_bytes, file_name="generated_image.png", mime="image/png")
        else:
            st.error("No image was generated. Please try again.")

with col2:
    if st.session_state.last_generated_image and os.path.exists(st.session_state.last_generated_image):
        st.subheader("Last Generated Image")
        st.image(st.session_state.last_generated_image, use_container_width=True)

