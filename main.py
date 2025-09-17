import subprocess
import os

def run_pipeline():
    '''
    input_wav = "./Input/Input_audio.wav"
    enroll_wav = "./Input/speaker_enrollment.wav"
    solospeech_out = "./Solospeech_Output/output_audio.wav"
    transcript_out = "./Transcription_Output/transcript.txt"
    '''

    input_wav = "./Input/test1.wav"
    enroll_wav = "./Input/test1_enroll.wav"
    solospeech_out = "./Solospeech_Output/output_audio_test1.wav"
    transcript_out = "./Transcription_Output/transcript_test1.txt"

    os.makedirs("./Solospeech_Output", exist_ok=True)
    os.makedirs("./Transcription_Output", exist_ok=True)

    # 1. Run SoloSpeech
    print("Running SoloSpeech...")
    subprocess.run([
        "python3", "SoloSpeech/scripts/test_v2.py",
        "--test-wav", input_wav,
        "--enroll-wav", enroll_wav,
        "--output-path", solospeech_out
    ], check=True)

    
    # 2. Run Whisper
    print("Running Whisper transcription...")
    subprocess.run([
        "python3", "whisper_runner.py",
        solospeech_out,
        transcript_out
    ], check=True)



if __name__ == "__main__":
    run_pipeline()
