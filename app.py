import streamlit as st
import google.generativeai as genai
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import os
import io
import tempfile
from dotenv import load_dotenv

# ============================================================
# LOAD API KEYS FROM .env FILE
# ============================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")
VISION_KEY = os.getenv("VISION_KEY")
VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")

# ============================================================
# CONFIGURE GOOGLE GEMINI (The Brain of the Bot)
# ============================================================
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ============================================================
# CONFIGURE AZURE COMPUTER VISION (Image Understanding)
# ============================================================
vision_client = ComputerVisionClient(
    VISION_ENDPOINT,
    CognitiveServicesCredentials(VISION_KEY)
)

# ============================================================
# SYSTEM PROMPT - Tells Gemini to act as a Travel Bot
# ============================================================
SYSTEM_PROMPT = """You are an expert Travel Destination Recommendation Bot. 
Your job is to recommend travel destinations based on user preferences.

When a user describes what they want (beach, mountains, adventure, budget, etc.), 
you should:
1. Recommend 2-3 destinations that match their preferences
2. For each destination, provide:
   - Why it matches their preferences
   - Best time to visit
   - Top 3 things to do there
   - Estimated budget (per person, approximate)
   - A practical travel tip

When a user shares an image description of a place, identify what kind of 
destination it is and suggest similar places they might enjoy.

Be friendly, enthusiastic, and helpful. Use emojis to make responses engaging.
Keep responses well-structured and easy to read."""


# ============================================================
# FUNCTION: Analyze Image using Azure Computer Vision
# ============================================================
def analyze_image(image_bytes):
    """
    Takes an image, sends it to Azure Computer Vision,
    and returns a text description of what's in the image.
    """
    try:
        # Send image to Azure Computer Vision
        image_stream = io.BytesIO(image_bytes)
        analysis = vision_client.analyze_image_in_stream(
            image_stream,
            visual_features=[
                VisualFeatureTypes.description,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.categories
            ]
        )

        # Get the description
        if analysis.description.captions:
            description = analysis.description.captions[0].text
        else:
            description = "No description available"

        # Get the tags (keywords)
        tags = [tag.name for tag in analysis.tags[:10]]
        tags_text = ", ".join(tags)

        result = f"Image Description: {description}\nKeywords: {tags_text}"
        return result

    except Exception as e:
        return f"Error analyzing image: {str(e)}"


# ============================================================
# FUNCTION: Convert Speech to Text using Azure Speech Service
# ============================================================
def speech_to_text(audio_file_path):
    """
    Takes an audio file, sends it to Azure Speech Service,
    and returns the spoken words as text.
    """
    try:
        # Set up the speech configuration
        speech_config = speechsdk.SpeechConfig(
            subscription=SPEECH_KEY,
            region=SPEECH_REGION
        )
        speech_config.speech_recognition_language = "en-US"

        # Set up the audio configuration from file
        audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)

        # Create the speech recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        # Recognize speech
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return "Sorry, I couldn't understand the audio. Please try again."
        else:
            return "Speech recognition failed. Please try again."

    except Exception as e:
        return f"Error in speech recognition: {str(e)}"


# ============================================================
# FUNCTION: Get Recommendation from Gemini
# ============================================================
def get_recommendation(user_input):
    """
    Takes the user's text input, sends it to Gemini,
    and returns a travel recommendation.
    """
    try:
        # Combine system prompt with user input
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_input}\n\nAssistant:"

        # Generate response from Gemini
        response = model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        return f"Error getting recommendation: {str(e)}"


# ============================================================
# STREAMLIT APP - The User Interface
# ============================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Travel Destination Bot",
    page_icon="✈️",
    layout="centered"
)

# --- App Title and Description ---
st.title("✈️ Travel Destination Recommendation Bot")
st.markdown("Welcome! I can help you find your perfect travel destination. "
            "Tell me what you're looking for using **text**, **voice**, or an **image**!")
st.markdown("---")

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input Mode Selection ---
st.sidebar.title("🔧 Input Mode")
input_mode = st.sidebar.radio(
    "How would you like to interact?",
    ["💬 Text", "🎤 Voice", "📷 Image"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About this Bot")
st.sidebar.markdown(
    "This bot uses:\n"
    "- **Google Gemini** for AI recommendations\n"
    "- **Azure Speech Service** for voice input\n"
    "- **Azure Computer Vision** for image analysis"
)

# ============================================================
# TEXT INPUT MODE
# ============================================================
if input_mode == "💬 Text":
    user_input = st.chat_input("Describe your ideal travel destination...")

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get and display bot response
        with st.chat_message("assistant"):
            with st.spinner("🌍 Finding the perfect destinations for you..."):
                response = get_recommendation(user_input)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================================
# VOICE INPUT MODE
# ============================================================
elif input_mode == "🎤 Voice":
    st.info("🎤 Upload a voice recording (.wav file) with your travel preferences.")

    audio_file = st.file_uploader(
        "Upload your voice recording",
        type=["wav"],
        key="voice_upload"
    )

    if audio_file is not None:
        # Play the audio for the user
        st.audio(audio_file)

        if st.button("🎤 Process Voice Input"):
            with st.spinner("🔊 Converting your speech to text..."):
                # Save the uploaded audio to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.getvalue())
                    tmp_path = tmp.name

                # Convert speech to text
                recognized_text = speech_to_text(tmp_path)

                # Clean up temp file
                os.unlink(tmp_path)

            # Show what was recognized
            st.success(f"🗣️ You said: *{recognized_text}*")

            # Add to chat and get recommendation
            st.session_state.messages.append(
                {"role": "user", "content": f"🎤 Voice: {recognized_text}"}
            )

            with st.chat_message("assistant"):
                with st.spinner("🌍 Finding the perfect destinations for you..."):
                    response = get_recommendation(recognized_text)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================================
# IMAGE INPUT MODE
# ============================================================
elif input_mode == "📷 Image":
    st.info("📷 Upload an image of a place you like, and I'll suggest similar destinations!")

    image_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if image_file is not None:
        # Display the uploaded image
        image = Image.open(image_file)
        st.image(image, caption="Your uploaded image", use_container_width=True)

        if st.button("🔍 Analyze Image & Get Recommendations"):
            with st.spinner("🔍 Analyzing your image..."):
                # Analyze the image with Azure Computer Vision
                image_bytes = image_file.getvalue()
                image_description = analyze_image(image_bytes)

            # Show what was detected in the image
            st.success(f"📸 Image Analysis:\n{image_description}")

            # Create a prompt combining image analysis with travel request
            image_prompt = (
                f"The user uploaded an image of a travel destination. "
                f"Here is the analysis of the image:\n{image_description}\n\n"
                f"Based on this image, suggest similar travel destinations "
                f"that the user might enjoy. Explain why each suggestion "
                f"matches the vibe and features of the uploaded image."
            )

            # Add to chat and get recommendation
            st.session_state.messages.append(
                {"role": "user", "content": f"📷 Image uploaded: {image_description}"}
            )

            with st.chat_message("assistant"):
                with st.spinner("🌍 Finding similar destinations for you..."):
                    response = get_recommendation(image_prompt)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
