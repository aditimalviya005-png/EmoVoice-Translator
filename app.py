import whisper
import librosa
import numpy as np
from googletrans import Translator
from gtts import gTTS
import gradio as gr
import os

# Whisper Model Load karein
model = whisper.load_model("base")
translator = Translator()

def detect_emotion(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        pitches = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        pitches = pitches[~np.isnan(pitches)]
        pitches = pitches[pitches > 0]
        
        if len(pitches) == 0:
            return "Neutral / Calm 😐"
            
        mean_pitch = np.mean(pitches)
        
        if mean_pitch > 220:
            return "Happy / Excited 😊🎉"
        elif mean_pitch > 160:
            return "Angry / High Energy 😠🔥"
        elif mean_pitch < 120:
            return "Sad / Low Energy 😢🌧️"
        else:
            return "Neutral / Calm 😐"
    except Exception as e:
        return "Neutral / Calm 😐"

def process_audio(audio_path, target_lang):
    if audio_path is None:
        return "Kripya audio upload ya record karein.", None, None, None

    result = model.transcribe(audio_path)
    original_text = result['text']
    
    detected_emotion = detect_emotion(audio_path)
    
    lang_map = {
        "Hindi": "hi", "English": "en", "Spanish": "es", 
        "French": "fr", "German": "de", "Tamil": "ta", 
        "Telugu": "te", "Marathi": "mr"
    }
    lang_code = lang_map.get(target_lang, "hi")
    
    translated = translator.translate(original_text, dest=lang_code)
    translated_text = translated.text
    
    tts = gTTS(text=translated_text, lang=lang_code, slow=False)
    output_audio_path = "translated_output.mp3"
    tts.save(output_audio_path)
    
    return original_text, detected_emotion, translated_text, output_audio_path

languages = ["Hindi", "English", "Spanish", "French", "German", "Tamil", "Telugu", "Marathi"]

interface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="Apna Audio Record karein ya File Upload karein"),
        gr.Dropdown(choices=languages, value="Hindi", label="Kis bhasha (Language) mein translate karna hai?")
    ],
    outputs=[
        gr.Textbox(label="Aapne kya bola (Original Text)"),
        gr.Textbox(label="Detected Emotion (Aawaz ka Bhaav)"),
        gr.Textbox(label="Translated Text (Anuvaad)"),
        gr.Audio(label="Translated Audio (Nayi Aawaz)", type="filepath")
    ],
    title="🎙️ EmoVoice: Audio-to-Audio Translator cum Emotion Detector",
    description="Yeh AI tool aapke audio ka emotion detect karega aur sath hi aapki pasandida bhasha mein use translate karke sunayega!"
)

interface.launch()
