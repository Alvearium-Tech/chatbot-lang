import os
import speech_recognition as sr
import whisper
import tempfile
import pyttsx3

temp_file = tempfile.mkdtemp()
save_path = os.path.join(temp_file, 'temp.wav')
print(f"This is save audio path: {save_path}")

listener = sr.Recognizer()

engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty('rate', 145)
engine.setProperty('voice', voices[2].id)

def talk(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    try:
        with sr.Microphone() as source:
            print("Say something...")
            listener.adjust_for_ambient_noise(source)
            audio = listener.listen(source)
            with open(save_path, "wb") as f:
                f.write(audio.get_wav_data())
    except Exception as e:
        print(e)
        return None  # Devolver None en caso de error

    return save_path

def recognize_audio(save_path):
    if save_path is None:
        return None  # Manejar caso donde no se pudo guardar el audio

    audio_model = whisper.load_model('medium')
    transcription = audio_model.transcribe(save_path, language='spanish', fp16=False)
    return transcription['text'] if 'text' in transcription else None

def main():
    audio_file = listen()
    if audio_file:
        response = recognize_audio(audio_file)
        if response:
            talk(response)
            print(response)
        else:
            print("No se pudo transcribir el audio.")

if __name__ == "__main__":
    main()
