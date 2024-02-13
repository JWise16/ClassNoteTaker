import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from openai import OpenAI
import io
import os

# Load the environment variables from the .env file
load_dotenv()

# Fetch the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate the presence of the API key
if OPENAI_API_KEY is None:
    st.error("OpenAI API key is missing. Please check your .env file.")
    st.stop()

# Initialize the OpenAI client with the API key
client = OpenAI(api_key=OPENAI_API_KEY)

def transcribe_audio(audio_bytes):
    """
    Transcribe audio using OpenAI's Whisper model.
    """
    # Convert the byte data to a file-like object
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    
    try:
        # Transcribe the audio using Whisper
        transcript_response = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        # Assuming the correct field to retrieve the text is 'text'
        # Adjust this attribute access based on the actual response structure
        transcript = transcript_response.text
        return transcript
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

def transcript_to_notes(transcript):
    try:
        completion = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a note taker in a college class. Turn the following class transcript into usable, well-organized notes."
                },
                {
                    "role": "user",
                    "content": transcript,
                },
            ]
        )
        notes = completion.choices[0].message.content
        return notes
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
st.title("Audio to Text with OpenAI Whisper and Note Generation")

# Record audio using 'mic_recorder' from the 'streamlit_mic_recorder' package
audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", key="whisper")

if audio:
    # Extract the audio bytes from the recorded data
    audio_bytes = audio['bytes']
    
    # Render the recorded audio for playback
    st.audio(audio_bytes)
    
    with st.spinner("Transcribing audio with Whisper..."):
        # Process the audio with Whisper to transcribe
        transcription = transcribe_audio(audio_bytes)
    
    if transcription:
        # Display the successful transcription
        st.success("Transcription successful!")
        st.text_area("Transcription:", value=transcription, height=100)
        
        # Generate notes from the transcription
        with st.spinner("Generating notes from transcription..."):
            notes = transcript_to_notes(transcription)

            if notes:
                st.success("Note generation successful!")
                st.text_area("Generated Notes:", value=notes, height=200)
            else:
                st.error("Failed to generate notes from the transcription.")
    else:
        # Handle failure in transcription
        st.error("Failed to transcribe audio.")
