import os
import shutil
import textwrap
import zipfile
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse

from moviepy.config import change_settings
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, AudioFileClip, CompositeAudioClip

import pysrt
from pydub import AudioSegment

import torch
from transformers import pipeline

from openai import OpenAI


change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16\\magick.exe"})
load_dotenv()
app = FastAPI()
client = OpenAI()
DEVICE = 0 if torch.cuda.is_available() else "cpu"


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

def decode_audio(inFile, outFile):
    if not outFile.endswith(".mp3"):
        outFile += ".mp3"
    AudioSegment.from_file(inFile).set_channels(1).export(outFile, format="mp3")

def transcribe_yoruba(file_path: str):
    model_name = "neoform-ai/whisper-medium-yoruba"
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        device=DEVICE,
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

def transcribe(file_path, source_language):
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="srt",
            language=source_language
        )
    with open("temp/transcript.srt", "w") as transcript_file:
        transcript_file.write(transcription)

def translate_srt(input_srt, output_srt, source_language, target_language):
    if source_language == 'fr':
        subs = pysrt.open(input_srt, encoding='latin1')
    else:
        subs = pysrt.open(input_srt, encoding='utf-8')

    url = 'https://translation.googleapis.com/language/translate/v2'

    for sub in subs:
        params = {
            'q': sub.text,
            'source': source_language,
            'target': target_language,
            'key': os.getenv('GCLOUD_API_KEY')
        }
        response = requests.get(url, params=params)
        content = response.json()['data']['translations'][0]['translatedText']

        sub.text = content.replace('&#39;', "'")

    subs.save(output_srt, encoding='utf-8')

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
    transcript = ' '.join([s.text for s in subs])
    with open('temp/transcript.txt', 'w') as f:
        f.write(transcript)

def add_tts_to_video(video_path, srt_path):
    def text_to_speech(text, output_filename):
        response = client.audio.speech.create(model="tts-1", voice="shimmer", input=text)
        with open(output_filename, "wb") as out:
            out.write(response.content)
    
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

def cap(video_file, source_language, target_language, dubbing):
    # Step 1: Decode audio from video
    audio_file_path = "temp/audio.mp3"
    decode_audio(video_file, audio_file_path)
    
    # Step 2: Transcribe audio to SRT
    if source_language == 'yo':
        transcribe_yoruba(audio_file_path)
    else:
        transcribe(audio_file_path, source_language)
    
    # Step 3: Translate the SRT
    translate_srt("temp/transcript.srt", "temp/translated_output.srt", source_language, target_language)
    
    if dubbing:
        # Step 4: Add TTS to video
        add_tts_to_video(video_file, "temp/translated_output.srt")
        # Step 5: Add captions to the video with TTS audio
        caption("temp/video_with_tts.mp4")
    else:
        # Step 4: Add captions to the original video
        caption(video_file)


@app.post("/process_video")
def process_video(file: UploadFile = File(...), source_language: str = Form(...), target_language: str = Form(...), dub: bool = Form(...)):
    if not os.path.exists('temp'):
        os.makedirs('temp')
    contents = file.file.read()
    file_path = os.path.join("temp", file.filename)
    with open(file_path, 'wb') as f:
        f.write(contents)

    cap(file_path, source_language, target_language, dub)

    zip_path = 'output.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write("temp/download.mp4", arcname='download.mp4')
        zipf.write('temp/transcript.txt', arcname='transcript.txt')

    shutil.rmtree('temp', ignore_errors=True)

    return FileResponse(zip_path, media_type='application/zip')
