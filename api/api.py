from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
import shutil
import torch
from transformers import pipeline
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, AudioFileClip, CompositeAudioClip

import pysrt
import textwrap
import requests
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
app = FastAPI()
client = OpenAI()


def clean_temp_folder():
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    os.makedirs('temp')
    
def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = ","):
    if seconds is not None:
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    else:
        return seconds

def create_srt(subtitles):
    srt_content = ""
    for i, (start, end, text) in enumerate(subtitles):
        start_time = format_timestamp(start, always_include_hours=True)
        end_time = format_timestamp(end, always_include_hours=True)
        srt_content += f"{i+1}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt_content

def text_to_speech(text, output_filename):
    response = client.audio.speech.create(model="tts-1", voice="shimmer", input=text)
    with open(output_filename, "wb") as out:
        out.write(response.content)

def decode_audio(inFile, outFile):
    if not outFile.endswith(".mp3"):
        outFile += ".mp3"
    AudioSegment.from_file(inFile).set_channels(1).export(outFile, format="mp3")

def transcribe_yoruba(file_path: str):
    # Initialize the model and arguments
    MODEL_NAME = "neoform-ai/whisper-medium-yoruba"
    device = 0 if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=device,
    )
    outputs = pipe(file_path, batch_size=8, generate_kwargs={"task": 'transcribe'}, return_timestamps=True)
    subtitles = []
    srt_content = ""
    timestamps = outputs["chunks"]
    subtitles = [
        (chunk['timestamp'][0], chunk['timestamp'][1], chunk['text'])
        for chunk in timestamps
    ]
    srt_content = create_srt(subtitles)
    with open("temp/transcript.srt", "w", encoding="utf-8") as file:
        file.write(srt_content)

def transcribe(file_path: str, language='french'):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="srt"
        )
    with open("temp/transcript.srt", "w") as transcript_file:
        transcript_file.write(transcription)
        

def caption(video_file):
    subs = pysrt.open('temp/translated_output.srt', encoding='utf-8')
    subtitles_list = [((s.start.ordinal / 1000, s.end.ordinal / 1000), s.text) for s in subs]
    video = VideoFileClip(video_file)

    def generator(txt, start, end):
        wrapped_txt = textwrap.fill(txt, width=30)  # Wrap the text if it's too long
        return (TextClip(wrapped_txt, font='DejaVu-Sans', fontsize=80, color='white', stroke_color='black', stroke_width=1.5)
                .set_start(start)
                .set_duration(end - start)
                .set_position(('center', 'bottom')))

    subtitles = [generator(txt, start, end) for ((start, end), txt) in subtitles_list]
    video = CompositeVideoClip([video] + subtitles)
    video.write_videofile("temp/download.mp4", codec='libx264', audio_codec='aac')

def translate_srt(input_srt, output_srt, target_language='en'):
    subs = pysrt.open(input_srt, encoding='latin-1')

    url = 'https://translation.googleapis.com/language/translate/v2'

    for sub in subs:
        params = {
            'q': sub.text,
            'target': target_language,
            'key': os.getenv('GCLOUD_API_KEY')
        }
        response = requests.get(url, params=params)
        sub.text = response.json()['data']['translations'][0]['translatedText']

    subs.save(output_srt, encoding='utf-8')

def add_tts_to_video(video_path, srt_path):
    subs = pysrt.open(srt_path, encoding='utf-8')
    tts_audios = []
    for i, sub in enumerate(subs):
        tts_audio_path = f"temp/tts_{i}.wav"
        text_to_speech(sub.text, tts_audio_path)
        audio_clip = AudioFileClip(tts_audio_path)
        duration = audio_clip.duration  # Get the actual duration of the TTS audio clip
        audio_clip = audio_clip.set_start(sub.start.ordinal / 1000).set_duration(duration)
        tts_audios.append(audio_clip)

    video = VideoFileClip(video_path)
    final_audio = CompositeAudioClip(tts_audios).set_duration(video.duration)
    final_video = video.set_audio(final_audio)
    final_video.write_videofile("temp/video_with_tts.mp4", codec='libx264', audio_codec='aac')

def cap(video_file, target_language, dubbing, source_language):
    # Step 1: Decode audio from video
    audio_file_path = "temp/audio.mp3"
    decode_audio(video_file, audio_file_path)
    
    # Step 2: Transcribe audio to SRT
    if source_language != 'Yoruba':
        transcribe(audio_file_path)
    else:
        transcribe_yoruba(audio_file_path)
    
    # Step 3: Translate the SRT
    translate_srt("temp/transcript.srt", "temp/translated_output.srt", target_language=target_language)
    
    if dubbing:
        # Step 4: Add TTS to video
        add_tts_to_video(video_file, "temp/translated_output.srt")
        # Step 5: Add captions to the video with TTS audio
        caption("temp/video_with_tts.mp4")
    else:
        # Step 4: Add captions to the original video
        caption(video_file)


@app.post("/process_video")
def process_video(file: UploadFile = File(...), target_language: str = Form(...), dub: bool = Form(...), source_language: str = Form(...)):
    clean_temp_folder()

    contents = file.file.read()
    file_path = os.path.join("temp", file.filename)
    with open(file_path, 'wb') as f:
        f.write(contents)

    cap(file_path, target_language, dub, source_language)

    return FileResponse("temp/download.mp4", media_type='video/mp4', filename="download.mp4")
