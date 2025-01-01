
import streamlit as st
from google.oauth2 import service_account
from google.cloud import speech
import io
import torch
from google.cloud import texttospeech
import numpy as np
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2 import Wav2Vec2Model
import librosa
from st_audiorec import st_audiorec
import scipy.io.wavfile as wav
import os
from datetime import datetime
from pydub import AudioSegment
from pathlib import Path
# from openai import OpenAI
import json
import time
import numpy as np
import base64
import openai


# Set OpenAI API key
openai.api_key = st.secrets["openai"] # Replace with your actual OpenAI API Key


# Set Google Cloud Service Account credentials
google_creds = st.secrets["google"]

credentials = service_account.Credentials.from_service_account_info(google_creds)


# Define constants
AUDIO_GAIN = 1.50  # Added constant for audio gain



# Initialize the TTS client with explicit credentials
tts_client = texttospeech.TextToSpeechClient(credentials=credentials)

# Initialize Speech client
speech_client = speech.SpeechClient(credentials=credentials)

processor = Wav2Vec2Processor.from_pretrained("models/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("models/wav2vec2-base")

ideal_embedding = torch.tensor(np.load("embeddings/ideal_embedding_part_1.npy"))


# Define ideal text
ideal_text = "ٱللَّهُ أَكْبَرُ ٱللَّهُ أَكْبَرُ"
ideal_text_meaning = "Allah is the Greatest, Allah is the Greatest"

def load_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Base styles */
:root {
    --primary-color: #2563eb;
    --secondary-color: #1d4ed8;
    --success-color: #059669;
    --warning-color: #d97706;
    --danger-color: #dc2626;
    --text-primary: #111827;
    --text-secondary: #4b5563;
    --bg-primary: #ffffff;
    --bg-secondary: #f3f4f6;
}

.stApp {
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
    background: var(--bg-secondary);
}

/* Header styles */
.app-header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    padding: 2rem 1rem;
    text-align: center;
    border-radius: 0 0 1.5rem 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.app-title {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.app-subtitle {
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.2rem;
    font-weight: 500;
    direction: rtl;
}

/* Card styles */
.card {
    background: var(--bg-primary);
    border-radius: 1rem;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;
}

.card:hover {
    transform: translateY(-2px);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--bg-secondary);
}

.card-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}

/* Button styles */
.button-container {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}

.button-primary {
    background-color: var(--primary-color);
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    border: none;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s ease;
    text-align: center;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.button-primary:hover {
    background-color: var(--secondary-color);
}

.button-danger {
    background-color: var(--danger-color);
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    border: none;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s ease;
}

/* Progress indicator */
.score-container {
    text-align: center;
    padding: 1.5rem;
    background: var(--bg-secondary);
    border-radius: 1rem;
    margin-bottom: 1.5rem;
}

.score-value {
    font-size: 3rem;
    font-weight: 700;
    color: var(--primary-color);
}

.score-label {
    color: var(--text-secondary);
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* Feedback section */
.feedback-section {
    background: var(--bg-secondary);
    border-radius: 1rem;
    padding: 1.5rem;
    margin-top: 1.5rem;
}

.feedback-item {
    background: white;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Status messages */
.success-msg {
    background-color: var(--success-color);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    margin-bottom: 1rem;
    animation: slideIn 0.3s ease;
}

.error-msg {
    background-color: var(--danger-color);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
    margin-bottom: 1rem;
    animation: slideIn 0.3s ease;
}

/* Animations */
@keyframes slideIn {
    from { transform: translateY(-10px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .app-title {
        font-size: 2rem;
    }
    
    .card {
        padding: 1rem;
    }
    
    .button-container {
        flex-direction: column;
    }
    
    .score-value {
        font-size: 2.5rem;
    }
}
</style>
""", unsafe_allow_html=True)



# Existing helper functions remain the same

def process_audio_for_visualization(audio_path, sr=None):
    """Process audio file and return simplified waveform data"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Reduce number of points more aggressively
        n = max(len(y) // 500, 1)  # Reduced to 500 points
        y_reduced = y[::n]
        time = np.linspace(0, len(y) / sr, len(y_reduced))
        
        # Normalize amplitude values
        y_reduced = y_reduced / np.max(np.abs(y_reduced))
        
        # Convert to regular Python types and round to 4 decimal places
        return [
            {
                "time": round(float(t), 4),
                "amplitude": round(float(a), 4)
            } 
            for t, a in zip(time, y_reduced)
        ]
    except Exception as e:
        print(f"Error in process_audio_for_visualization: {str(e)}")
        return []

def add_waveform_to_app():
    """Add simplified waveform visualization to Streamlit app"""
    try:
        if not st.session_state.get('audio_file') or not os.path.exists(st.session_state['audio_file']):
            return
        
        # Process user's audio
        user_waveform = process_audio_for_visualization(st.session_state['audio_file'])
        if not user_waveform:
            st.error("Could not process user audio for visualization")
            return
            
        # Process expert's audio
        expert_audio_path = r"audio_files/qari_part_1.mp3"
        expert_waveform = process_audio_for_visualization(expert_audio_path)
        if not expert_waveform:
            st.error("Could not process expert audio for visualization")
            return

        # Create safer combined data
        min_length = min(len(user_waveform), len(expert_waveform))
        combined_data = []
        
        for i in range(min_length):
            point = {
                'time': user_waveform[i]['time'],
                'userAmplitude': user_waveform[i]['amplitude'],
                'expertAmplitude': expert_waveform[i]['amplitude']
            }
            combined_data.append(point)

        # Verify data is JSON serializable
        try:
            json.dumps(combined_data)
        except TypeError as e:
            st.error(f"Data serialization error: {str(e)}")
            return

        # Create visualization
        st.components.v1.html(
            f"""
            <div style="padding: 10px;">
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <div id="waveformChart"></div>
                <script>
                    const data = {json.dumps(combined_data)};
                    
                    const traces = [
                        {{
                            x: data.map(d => d.time),
                            y: data.map(d => d.userAmplitude),
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Your Recording',
                            line: {{ color: '#8884d8' }}
                        }},
                        {{
                            x: data.map(d => d.time),
                            y: data.map(d => d.expertAmplitude),
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Expert Recording',
                            line: {{ color: '#82ca9d' }}
                        }}
                    ];
                    
                    const layout = {{
                        title: 'Waveform Comparison',
                        xaxis: {{ title: 'Time (s)' }},
                        yaxis: {{ title: 'Amplitude' }},
                        height: 400,
                        margin: {{ t: 30, r: 20, b: 40, l: 50 }}
                    }};
                    
                    Plotly.newPlot('waveformChart', traces, layout);
                </script>
            </div>
            """,
            height=450
        )
        
    except Exception as e:
        st.error(f"Error in waveform visualization: {str(e)}")
        print(f"Detailed error: {str(e)}")  # For debugging

def extract_feedback_text(feedback_text):
    """Extract only the main text content from the feedback"""
    try:
        # Split into sections using known markers
        content_parts = feedback_text.split("Here's the feedback")
        
        if len(content_parts) > 1:
            # Take the part after "Here's the feedback"
            main_content = content_parts[1]
            
            # Remove all bullet points and their variations
            main_content = main_content.replace('•', '')
            main_content = main_content.replace('*', '')
            
            # Remove section headers
            sections_to_remove = [
                "Talaffuz (Pronunciation):",
                "Waqt aur Lehja (Timing):",
                "Behtar Karne Ke Liye Mashwaray:"
            ]
            
            for section in sections_to_remove:
                main_content = main_content.replace(section, '')
            
            # Clean up extra whitespace and newlines
            lines = [line.strip() for line in main_content.split('\n') if line.strip()]
            cleaned_text = ' '.join(lines)
            
            # Remove any extra spaces
            cleaned_text = ' '.join(cleaned_text.split())
            
            return cleaned_text
            
        return feedback_text  # Return original if splitting fails
        
    except Exception as e:
        print(f"Error in extract_feedback_text: {str(e)}")
        return feedback_text

def generate_feedback_audio(feedback_text):
    """Generate audio file from feedback text using Google Cloud TTS"""
    try:
        # Create a directory for feedback audio if it doesn't exist
        if not os.path.exists("feedback_audio"):
            os.makedirs("feedback_audio")
            
        # Extract only the actual text content
        cleaned_feedback = extract_feedback_text(feedback_text)
        print("Cleaned feedback:", cleaned_feedback)  # For debugging
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = Path("feedback_audio") / f"feedback_{timestamp}.mp3"
        
        # Set up the synthesis input with cleaned text
        synthesis_input = texttospeech.SynthesisInput(text=cleaned_feedback)
        
        # Configure voice parameters
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-IN",
            name="en-IN-Wavenet-B"
        )
        
        # Configure audio parameters
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            effects_profile_id=['small-bluetooth-speaker-class-device'],
            speaking_rate=1,
            pitch=1,
        )
        
        # Generate the speech using the global tts_client
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Save the audio file
        with open(str(audio_path), "wb") as out:
            out.write(response.audio_content)
            
        return str(audio_path)
    
    except Exception as e:
        st.error(f"Error generating audio feedback: {str(e)}")
        return None
def enhance_audio(audio_data, gain=AUDIO_GAIN):
    """Enhance audio quality with noise reduction and gain adjustment"""
    audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
    audio_data = audio_data * gain
    noise_threshold = 0.01
    audio_data[np.abs(audio_data) < noise_threshold] = 0
    return audio_data

def get_audio_embedding(audio_file_path):
    """Generate audio embedding using Wav2Vec2"""
    audio_input, _ = librosa.load(audio_file_path, sr=16000)
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model(inputs.input_values).last_hidden_state.mean(dim=1).squeeze()
    return embedding

def calculate_similarity(embedding1, embedding2):
    """Calculate similarity score with adjusted scoring for 70-89 range"""
    # Calculate base similarity using cosine similarity
    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
    base_score = similarity.item() * 100
    
    # Adjust score for 70-89 range
    if 70 <= base_score <= 89:
        adjusted_score = base_score - 20
    else:
        adjusted_score = base_score
        
    return adjusted_score
import librosa
import numpy as np
from typing import Dict, List, Tuple

def analyze_audio_features(audio_path: str, reference_path: str) -> Dict:
    """
    Analyze specific audio features for each word in the Azaan
    """
    # Load audio files
    user_audio, sr = librosa.load(audio_path, sr=16000)
    ref_audio, _ = librosa.load(reference_path, sr=16000)
    
    # Define Azaan words and their expected durations/features
    azaan_words = {
        "Allahu": {
            "expected_duration": 1.2,
            "pitch_range": (120, 150),
            "energy_threshold": 0.7
        },
        "Akbar": {
            "expected_duration": 1.0,
            "pitch_range": (100, 130),
            "energy_threshold": 0.6
        }
    }
    
    # Extract features for each word
    word_analysis = {}
    
    for word, expected in azaan_words.items():
        # Get word boundaries using onset detection
        onset_frames = librosa.onset.onset_detect(y=user_audio, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Extract pitch (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=user_audio, sr=sr)
        pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes)*0.1])
        
        # Calculate duration
        word_duration = onset_times[1] - onset_times[0] if len(onset_times) > 1 else 0
        
        # Get energy/volume
        rms = librosa.feature.rms(y=user_audio)[0]
        energy = np.mean(rms)
        
        # Compare with expected values
        word_analysis[word] = {
            "duration_match": abs(word_duration - expected["expected_duration"]),
            "pitch_match": expected["pitch_range"][0] <= pitch_mean <= expected["pitch_range"][1],
            "energy_match": energy >= expected["energy_threshold"]
        }
    
    return word_analysis


def generate_feedback_with_llm(user_transcription, ideal_text, similarity_score, word_analysis: Dict):
    """Generate detailed feedback using LLM with audio analysis data"""
    
    # Create a detailed analysis summary for LLM
    analysis_summary = []
    for word, analysis in word_analysis.items():
        word_details = {
            "word": word,
            "duration_difference": f"{analysis['duration_match']:.2f} seconds",
            "pitch_in_range": "Yes" if analysis['pitch_match'] else "No",
            "energy_sufficient": "Yes" if analysis['energy_match'] else "No"
        }
        analysis_summary.append(word_details)

    prompt = f"""
    You are an expert Azaan teacher. Analyze the student's recitation and provide clear, simple feedback in English.

    Recitation Details:
    User's recitation: {user_transcription}
    Correct text: {ideal_text}
    Similarity score: {similarity_score:.2f}%

    Technical analysis for each word:
    {json.dumps(analysis_summary, indent=2)}

    Provide feedback in these sections:

    1. Voice and Pronunciation:
    - Specific guidance for each word
    - Voice pitch analysis
    - Word accuracy

    2. Timing and Style:
    - Duration analysis for each word
    - Pause accuracy
    - Voice modulation details

    3. Improvement Tips:
    - Word-specific advice
    - Correct pronunciation guide
    - Voice volume and duration guidance

    Note: 
    - Provide specific instructions based on the technical analysis
    - Use simple, clear English
    - Be precise and direct
    - Avoid using stars or bullet points
    - Keep feedback encouraging but focused on improvement
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert Azaan teacher who provides technical feedback.
                    Give clear, simple feedback in English.
                    Focus on pronunciation, duration, pitch, and energy for each word.
                    Be direct and specific with suggestions.
                    Avoid using stars, bullet points, or complex terminology."""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        feedback = response.choices[0].message["content"].strip()
        return feedback
    except Exception as e:
        error_msg = str(e)
        return f"Error generating feedback: {error_msg}"

def record_audio_with_progress():
    """Record audio with a progress bar showing recording duration"""
    duration = 6  # Recording duration in seconds
    sample_rate = 48000
    channels = 1
    
    # Create a placeholder for the progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize recording
    audio_data = np.zeros((int(duration * sample_rate), channels), dtype=np.float32)
    stream = sd.InputStream(samplerate=sample_rate, channels=channels)
    stream.start()
    
    # Record with progress bar
    start_time = time.time()
    for i in range(int(duration)):
        status_text.text(f"Recording: {duration - i} seconds remaining...")
        chunk, _ = stream.read(sample_rate)
        audio_data[i*sample_rate:(i+1)*sample_rate] = chunk
        progress_bar.progress((i + 1) / duration)
        time.sleep(max(0, (start_time + i + 1) - time.time()))
    
    stream.stop()
    stream.close()
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Process audio
    audio_data = enhance_audio(audio_data.flatten(), AUDIO_GAIN)
    audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
    
    # Save recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recordings/audio_{timestamp}.wav"
    wav.write(filename, sample_rate, audio_data)
    
    return filename

# def transcribe_and_validate(audio_file_path, ideal_text):
#     try:
#         # Convert WAV to MP3
#         mp3_path = audio_file_path.replace('.wav', '.mp3')
#         audio = AudioSegment.from_wav(audio_file_path)
#         audio.export(mp3_path, format="mp3")
        
#         with io.open(mp3_path, 'rb') as f:
#             audio_content = f.read()
        
#         audio = speech.RecognitionAudio(content=audio_content)
#         config = speech.RecognitionConfig(
#             encoding=speech.RecognitionConfig.AudioEncoding.MP3,
#             sample_rate_hertz=48000,
#             language_code="ar"
#         )
        
#         response = speech_client.recognize(config=config, audio=audio)
#         transcription = " ".join(result.alternatives[0].transcript for result in response.results)
        
#         # Clean up temporary MP3 file
#         if os.path.exists(mp3_path):
#             os.remove(mp3_path)
            
#         content = f"""
#             You are an expert in validating the Azaan (the call to prayer). Below is the correct structure of the Azaan. 
#             Compare the transcription provided with this structure to determine if it contains all essential phrases in the correct order.
#             Validation Guidelines:
#             - Validate the Azaan as "VALIDATED" if it contains all essential phrases in the correct sequence, even if there are minor spelling, diacritic, or punctuation differences.
#             - Specifically, ignore small differences such as:
#                 - Missing or extra diacritics (e.g., "ا" vs. "أ" or "حي على الصلاه" vs. "حي على الصلاة").
#                 - Minor spelling variations
#                 - Punctuation or slight variations in commonly understood words and phrases.
#             Correct Azaan Structure:
#             "{ideal_text}"
#             Transcribed Azaan:
#             "{transcription}"
#             Conclude with "Validation Status: VALIDATED" if the Azaan matches the correct structure, or "Validation Status: INVALIDATED" if it does not.
#         """
        
#         completion = groq_client.chat.completions.create(
#             model="llama3-70b-8192",
#             messages=[{"role": "user", "content": content}],
#             temperature=0,
#             max_tokens=512,
#         )
#         validation_feedback = completion.choices[0].message.content
        
#         # If validation is successful, calculate similarity
#         if "Validation Status: VALIDATED" in validation_feedback:
#             user_embedding = get_audio_embedding(audio_file_path)
#             similarity_score = calculate_similarity(user_embedding, ideal_embedding)
            
#             # Get detailed audio analysis
#             word_analysis = analyze_audio_features(
#                 audio_file_path,
#                 "audio_files\\qari_part_1.mp3"  # reference audio
#             )
            
#             # Generate comprehensive feedback using both analysis and LLM
#             feedback = generate_feedback_with_llm(
#                 transcription, 
#                 ideal_text, 
#                 similarity_score,
#                 word_analysis
#             )
            
#             return transcription, validation_feedback, similarity_score, feedback
            
#         return transcription, validation_feedback, None, None
        
    # except Exception as e:
    #     st.error(f"Error during processing: {str(e)}")
    #     return None, None, None, None
        
    # except Exception as e:
    #     st.error(f"Error during processing: {str(e)}")
    #     return None, None, None, None
# Updated Main Function with Persian Theme


def transcribe_and_validate(audio_file_path, ideal_text):
    try:
        # Convert WAV to MP3
        mp3_path = audio_file_path.replace('.wav', '.mp3')
        audio = AudioSegment.from_wav(audio_file_path)
        audio.export(mp3_path, format="mp3")
        
        with io.open(mp3_path, 'rb') as f:
            audio_content = f.read()
        
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MP3,
            sample_rate_hertz=48000,
            language_code="ar"
        )
        
        response = speech_client.recognize(config=config, audio=audio)
        transcription = " ".join(result.alternatives[0].transcript for result in response.results)
        
        # Clean up temporary MP3 file
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
            
        content = f"""
            You are an expert in validating the Azaan (the call to prayer). Below is the correct structure of the Azaan. 
            Compare the transcription provided with this structure to determine if it contains all essential phrases in the correct order.
            Validation Guidelines:
            - Validate the Azaan as "VALIDATED" if it contains all essential phrases in the correct sequence, even if there are minor spelling, diacritic, or punctuation differences.
            - Specifically, ignore small differences such as:
                - Missing or extra diacritics (e.g., "ا" vs. "أ" or "حي على الصلاه" vs. "حي على الصلاة").
                - Minor spelling variations
                - Punctuation or slight variations in commonly understood words and phrases.
            Correct Azaan Structure:
            "{ideal_text}"
            Transcribed Azaan:
            "{transcription}"
            Conclude with "Validation Status: VALIDATED" if the Azaan matches the correct structure, or "Validation Status: INVALIDATED" if it does not.
        """
        
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=512,
        )
        validation_feedback = completion.choices[0].message.content
        
        # If validation is successful, calculate similarity
        if "Validation Status: VALIDATED" in validation_feedback:
            user_embedding = get_audio_embedding(audio_file_path)
            similarity_score = calculate_similarity(user_embedding, ideal_embedding)
            
            # Get detailed audio analysis
            reference_path=r"audio_files/qari_part_1.mp3"
            word_analysis = analyze_audio_features(
                audio_file_path,
                reference_path # reference audio
            )
            
            # Generate comprehensive feedback using both analysis and LLM
            feedback = generate_feedback_with_llm(
                transcription, 
                ideal_text, 
                similarity_score,
                word_analysis
            )
            
            return transcription, validation_feedback, similarity_score, feedback
            
        return transcription, validation_feedback, None, None
        
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        return None, None, None, None
        
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        return None, None, None, None

def main():
    st.set_page_config(
        page_title="Azaan Trainer",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS with Persian/Masjid-inspired theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap');
    
    :root {
        --primary-color: #1F4C6B;
        --secondary-color: #C3934B;
        --accent-color: #E6B17E;
        --background-color: #F7F3E9;
        --text-color: #2C3E50;
        --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stApp {
        background-color: var(--background-color);
        font-family: 'Amiri', serif;
    }

    .app-header {
        background: linear-gradient(135deg, var(--primary-color), #2C3E50);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--card-shadow);
    }

    .app-title {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, var(--accent-color), #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .app-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0.5rem 0;
    }

    .arabic-text {
        font-family: 'Amiri', serif;
        font-size: 2rem;
        direction: rtl;
        margin: 1rem 0;
        color: var(--secondary-color);
    }

    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(195, 147, 75, 0.2);
        transition: transform 0.2s ease;
    }

    .card:hover {
        transform: translateY(-2px);
    }

    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 0.5rem;
    }

    .card-title {
        font-size: 1.3rem;
        margin: 0 0 0 0.5rem;
        color: var(--primary-color);
    }

    .stButton button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        margin: 0.5rem 0;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 76, 107, 0.2);
    }

    .score-container {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
    }

    .score-value {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .score-label {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    .feedback-item {
        background-color: rgba(195, 147, 75, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid var(--secondary-color);
    }

    .help-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
    }

    .help-item {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding: 0.5rem;
        border-radius: 8px;
        background-color: rgba(31, 76, 107, 0.05);
    }

    .help-number {
        background-color: var(--primary-color);
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-size: 0.9rem;
    }

    .stAudioRecorderControl {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
        border-radius: 25px !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }

    .stAudioRecorderControl:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(31, 76, 107, 0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Enhanced Header with Arabic Styling
    st.markdown(f"""
    <div class="app-header">
        <h1 class="app-title">Azaan Trainer</h1>
        <p class="app-subtitle">Perfect Your Recitation</p>
        <div class="arabic-text">{ideal_text}</div>
        <p class="app-subtitle">{ideal_text_meaning}</p>
    </div>
    """, unsafe_allow_html=True)

    # How to Use Section
    with st.expander("❓ How to Use"):
        st.markdown("""
        <div class="help-container">
            <div class="help-item">
                <div class="help-number">1</div>
                <div>Watch the expert demonstration video carefully</div>
            </div>
            <div class="help-item">
                <div class="help-number">2</div>
                <div>Listen to the reference audio to understand proper pronunciation</div>
            </div>
            <div class="help-item">
                <div class="help-number">3</div>
                <div>Click the microphone icon and recite the phrase (6 seconds)</div>
            </div>
            <div class="help-item">
                <div class="help-number">4</div>
                <div>Wait for the recording to complete and automatic analysis</div>
            </div>
            <div class="help-item">
                <div class="help-number">5</div>
                <div>Review your score and feedback to improve</div>
            </div>
            <div class="help-item">
                <div class="help-number">6</div>
                <div>Practice until you achieve 90% or higher similarity</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Expert demonstration card
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span style="font-size: 2rem;">📹</span>
            <h2 class="card-title">Expert Demonstration</h2>
        </div>
    """, unsafe_allow_html=True)
    st.video("https://drive.google.com/file/d/1zBLhixpPCwSHF2_9a7fNDVoVRktV9OZu/view?usp=sharing")
    st.markdown("</div>", unsafe_allow_html=True)

    # Expert audio card
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span style="font-size: 2rem;">🎵</span>
            <h2 class="card-title">Reference Audio</h2>
        </div>
    """, unsafe_allow_html=True)
    st.audio(r"audio_files\qari_part_1.mp3")
    st.markdown("</div>", unsafe_allow_html=True)

    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state['analysis_complete'] = False
    if 'audio_file' not in st.session_state:
        st.session_state['audio_file'] = None
    if 'audio_feedback_path' not in st.session_state:
        st.session_state['audio_feedback_path'] = None


    # How to Use Section




    # Recording Section with Web Audio Recorder
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span style="font-size: 2rem;">🎙️</span>
            <h2 class="card-title">Recording Controls</h2>
        </div>
        
    """, unsafe_allow_html=True)

    # Use st_audiorec for recording
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        # Save and process the recording
        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"recordings/audio_{timestamp}.wav"
        
        with open(temp_path, 'wb') as f:
            f.write(wav_audio_data)
        
        st.session_state['audio_file'] = temp_path

        # Display recorded audio
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <span style="font-size: 2rem;">🎵</span>
                <h2 class="card-title">Your Recording</h2>
            </div>
        """, unsafe_allow_html=True)
        st.audio(wav_audio_data, format='audio/wav')
        
        # Waveform visualization
        st.markdown("""
            <div class="card-header">
                <span style="font-size: 2rem;">📊</span>
                <h2 class="card-title">Waveform Comparison</h2>
            </div>
        """, unsafe_allow_html=True)
        add_waveform_to_app()

        # Analysis Section
        if not st.session_state['analysis_complete']:
            with st.spinner("Analyzing your pronunciation..."):
                transcription, validation_feedback, similarity_score, feedback = transcribe_and_validate(
                    st.session_state['audio_file'],
                    ideal_text
                )
                
                if transcription and validation_feedback:
                    if "Validation Status: VALIDATED" in validation_feedback:
                        st.markdown(f"""
                        <div class="score-container">
                            <div class="score-value">{similarity_score-20:.1f}%</div>
                            <div class="score-label">Similarity Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="card">
                            <div class="card-header">
                                <span style="font-size: 2rem;">📝</span>
                                <h2 class="card-title">Detailed Feedback</h2>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        feedback_content = "🌟 Excellent work! Your pronunciation is reverent and accurate." if similarity_score >= 90 else feedback
                        st.markdown(f"""
                        <div class="feedback-item" style="background-color: {'rgba(46, 204, 113, 0.1)' if similarity_score >= 90 else 'rgba(231, 76, 60, 0.1)'}; border-left-color: {'#2ecc71' if similarity_score >= 90 else '#e74c3c'};">
                            {feedback_content}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Audio feedback
                        audio_path = generate_feedback_audio(feedback_content)
                        if audio_path:
                            st.session_state['audio_feedback_path'] = audio_path
                            st.markdown("""
                            <div class="card">
                                <div class="card-header">
                                    <span style="font-size: 2rem;">🔊</span>
                                    <h2 class="card-title">Audio Feedback</h2>
                                </div>
                            """, unsafe_allow_html=True)
                            st.audio(audio_path, format="audio/mp3")
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="feedback-item" style="background-color: rgba(231, 76, 60, 0.1); border-left-color: #e74c3c;">
                            ⚠️ Please try recording again. Focus on pronouncing each word clearly.
                        </div>
                        """, unsafe_allow_html=True)
            
            st.session_state['analysis_complete'] = True

        # Add clear recording button
        # if st.button("🗑️ Clear Recording"):
        #     if st.session_state['audio_file'] and os.path.exists(st.session_state['audio_file']):
        #         os.remove(st.session_state['audio_file'])
        #     st.session_state['audio_file'] = None
        #     st.session_state['analysis_complete'] = False
        #     st.session_state['audio_feedback_path'] = None
        #     st.rerun()

    # Create necessary directories
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("feedback_audio", exist_ok=True)

if __name__ == "__main__":
    main()
