import streamlit as st
import ollama
import time
import json
import os
from datetime import datetime
from openai import OpenAI
import speech_recognition as sr
import tempfile
from pydub import AudioSegment

# List of available models
MODELS = [
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",  # OpenAI models
    "llama3.1:8b", "gemma2:2b", "mistral-nemo:latest", "phi3:latest",  # Ollama models
]

# Speech aspects to analyze
SPEECH_ASPECTS = [
    "Content", "Structure", "Delivery", "Body Language", "Voice Modulation",
    "Engagement", "Confidence", "Clarity", "Persuasiveness", "Time Management"
]

def get_ai_response(messages, model):
    if model.startswith("gpt-"):
        return get_openai_response(messages, model)
    else:
        return get_ollama_response(messages, model)

def get_openai_response(messages, model):
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, 0, 0

def get_ollama_response(messages, model):
    try:
        response = ollama.chat(
            model=model,
            messages=messages
        )
        return response['message']['content'], response['prompt_eval_count'], response['eval_count']
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, 0, 0

def stream_response(messages, model):
    if model.startswith("gpt-"):
        return stream_openai_response(messages, model)
    else:
        return stream_ollama_response(messages, model)

def stream_openai_response(messages, model):
    client = OpenAI()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        return stream
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def stream_ollama_response(messages, model):
    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )
        return stream
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def save_conversation(messages, filename):
    conversation = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    
    os.makedirs('conversations', exist_ok=True)
    file_path = os.path.join('conversations', filename)
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                conversations = json.load(f)
        else:
            conversations = []
    except json.JSONDecodeError:
        conversations = []
    
    conversations.append(conversation)
    
    with open(file_path, 'w') as f:
        json.dump(conversations, f, indent=2)

def load_conversations(uploaded_file):
    if uploaded_file is not None:
        try:
            conversations = json.loads(uploaded_file.getvalue().decode("utf-8"))
            return conversations
        except json.JSONDecodeError:
            st.error(f"Error decoding the uploaded file. The file may be corrupted or not in JSON format.")
            return []
    else:
        st.warning("No file was uploaded.")
        return []

def transcribe_audio(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from speech recognition service; {e}"

def main():
    st.set_page_config(layout="wide")
    st.title("Public Speaking Trainer")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "token_count" not in st.session_state:
        st.session_state.token_count = {"prompt": 0, "completion": 0}

    if "user_name" not in st.session_state:
        st.session_state.user_name = "Speaker"

    st.session_state.user_name = st.text_input("Enter your name:", value=st.session_state.user_name)

    st.sidebar.title("Speech Analysis Configuration")
    model = st.sidebar.selectbox("Choose a model", MODELS)

    aspects_to_analyze = st.sidebar.multiselect("Aspects to Analyze", SPEECH_ASPECTS, default=SPEECH_ASPECTS[:5])

    custom_instructions = st.sidebar.text_area("Custom Instructions", 
        f"""You are an expert Public Speaking Trainer AI. Your role is to analyze speeches and provide constructive feedback to help users improve their public speaking skills. Focus on the following aspects:

{', '.join(aspects_to_analyze)}

When analyzing a speech:
1. Provide specific, actionable feedback on each selected aspect
2. Highlight strengths and areas for improvement
3. Offer practical tips and exercises to enhance speaking skills
4. Consider the context and purpose of the speech when giving feedback
5. Be encouraging and supportive while maintaining honesty

When interacting with the user:
- Explain public speaking concepts clearly if requested
- Answer follow-up questions about the feedback or public speaking techniques
- Provide examples or demonstrations when helpful
- Adapt your feedback style to the user's experience level and goals

Remember, your goal is to help users become more confident and effective public speakers through constructive feedback and guidance.""")

    theme = st.sidebar.selectbox("Choose a theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.token_count = {"prompt": 0, "completion": 0}

    st.sidebar.subheader("Conversation Management")
    save_name = st.sidebar.text_input("Save conversation as:", "speech_feedback_session.json")
    if st.sidebar.button("Save Conversation"):
        save_conversation(st.session_state.messages, save_name)
        st.sidebar.success(f"Conversation saved to conversations/{save_name}")

    st.sidebar.subheader("Load Conversation")
    uploaded_file = st.sidebar.file_uploader("Choose a file to load conversations", type=["json"], key="conversation_uploader")
    
    if uploaded_file is not None:
        try:
            conversations = load_conversations(uploaded_file)
            if conversations:
                st.sidebar.success(f"Loaded {len(conversations)} conversations from the uploaded file")
                selected_conversation = st.sidebar.selectbox(
                    "Select a conversation to load",
                    range(len(conversations)),
                    format_func=lambda i: conversations[i]['timestamp']
                )
                if st.sidebar.button("Load Selected Conversation"):
                    st.session_state.messages = conversations[selected_conversation]['messages']
                    st.sidebar.success("Conversation loaded successfully!")
            else:
                st.sidebar.error("No valid conversations found in the uploaded file.")
        except Exception as e:
            st.sidebar.error(f"Error loading conversations: {str(e)}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    uploaded_audio = st.file_uploader("Upload your speech audio file", type=["wav", "mp3", "ogg"])
    
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")
        
        if st.button("Analyze Speech"):
            with st.spinner("Transcribing and analyzing your speech..."):
                # Convert uploaded file to wav for compatibility with speech_recognition
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    if uploaded_audio.type == "audio/wav":
                        tmp_file.write(uploaded_audio.getvalue())
                    else:
                        audio = AudioSegment.from_file(uploaded_audio)
                        audio.export(tmp_file.name, format="wav")
                    
                    transcription = transcribe_audio(tmp_file.name)
                
                os.unlink(tmp_file.name)  # Delete the temporary file
                
                prompt = f"Analyze the following speech transcription and provide feedback on {', '.join(aspects_to_analyze)}:\n\n{transcription}"
                st.session_state.messages.append({"role": "user", "content": f"{st.session_state.user_name}: {prompt}"})
                
                ai_messages = [
                    {"role": "system", "content": custom_instructions},
                    {"role": "user", "content": prompt}
                ]

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in stream_response(ai_messages, model):
                        if chunk:
                            if model.startswith("gpt-"):
                                full_response += chunk.choices[0].delta.content or ""
                            else:
                                full_response += chunk['message']['content']
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.05)
                    message_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

                _, prompt_tokens, completion_tokens = get_ai_response(ai_messages, model)
                st.session_state.token_count["prompt"] += prompt_tokens
                st.session_state.token_count["completion"] += completion_tokens

    if prompt := st.chat_input("Ask for public speaking advice or clarification on feedback:"):
        st.session_state.messages.append({"role": "user", "content": f"{st.session_state.user_name}: {prompt}"})
        with st.chat_message("user"):
            st.markdown(f"{st.session_state.user_name}: {prompt}")

        ai_messages = [
            {"role": "system", "content": custom_instructions},
            {"role": "system", "content": "Provide helpful advice on public speaking or clarify the feedback given on the user's speech."},
        ] + st.session_state.messages

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in stream_response(ai_messages, model):
                if chunk:
                    if model.startswith("gpt-"):
                        full_response += chunk.choices[0].delta.content or ""
                    else:
                        full_response += chunk['message']['content']
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        _, prompt_tokens, completion_tokens = get_ai_response(ai_messages, model)
        st.session_state.token_count["prompt"] += prompt_tokens
        st.session_state.token_count["completion"] += completion_tokens

    st.sidebar.subheader("Token Usage")
    st.sidebar.write(f"Prompt tokens: {st.session_state.token_count['prompt']}")
    st.sidebar.write(f"Completion tokens: {st.session_state.token_count['completion']}")
    st.sidebar.write(f"Total tokens: {sum(st.session_state.token_count.values())}")

if __name__ == "__main__":
    main()
