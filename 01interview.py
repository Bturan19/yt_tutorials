import os
import streamlit as st
from pathlib import Path
import speech_recognition as sr
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI

AUDIO_FILE = Path("microphone-results.wav")
os.environ["OPENAI_API_KEY"] = "your-key-here"

def record_audio(file_path):
    """Records audio to the specified file path and returns whether the recording was successfull"""
    st.write("Recording started... please speak.")
    r = sr.Recognizer()
    r.energy_threshold = 400
    try:
        with sr.Microphone() as source:
            audio = r.listen(source, timeout=30, phrase_time_limit=15)
        with open(file_path, "wb") as f:
            f.write(audio.get_wav_data())
        st.write("Recording is finished.")
        return True
    except Exception as e:
        st.error(f"An error occured while recording: {e}")
        return False
    
def transcribe_audio(file_path):
    """Transcribes the audio"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        st.warning("Could not understand the audio")
    return None

def get_response_from_gpt(prompt):
    """Generates a response from GPT"""
    llm = ChatOpenAI(model="gpt-4-turbo-preview",
                     temperature=0,
                     streaming=True,
                     callback_manager=CallbackManager([StreamlitCallbackHandler(st.container())])
                    )
    response = llm.predict(prompt)
    return response

def app():
    st.title("Speech to Text / AI Interview Helper")

    if 'history' not in st.session_state:
        st.session_state["history"] = []

    #Let's display previous conversations
    if st.session_state["history"]:
        with st.expander("View previous conversations"):
            for entry in st.session_state['history']:
                st.markdown(f">{entry}")

    # Record and process voice input
    if st.button("Start & Stop Recording"):
        if record_audio(str(AUDIO_FILE)):
            user_input = transcribe_audio(str(AUDIO_FILE))
            if user_input:
                response = get_response_from_gpt(user_input)
                if response:
                    conversation = f"You: {user_input}\nLLM:{response}"
                    st.session_state['history'].append(conversation)
                else:
                    st.write("LLM did not provide a response")
            else:
                st.write("Transcription unsuccessful")
    
    if st.button("Clear Conversation"):
        st.session_state['history'] = []

if __name__ == "__main__":
    app()