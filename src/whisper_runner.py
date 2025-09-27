import whisper
import sys

def run_whisper(input_wav: str, output_txt: str):
    # Load model (choose "turbo", "base", etc.)
    model = whisper.load_model("turbo")

    # Load and preprocess audio
    audio = whisper.load_audio(input_wav)
    audio = whisper.pad_or_trim(audio)

    # Mel spectrogram
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # Detect language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # Decode
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # Save to file
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(result.text)

    print("Transcription saved to", output_txt)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python whisper_runner.py <input_wav> <output_txt>")
        sys.exit(1)

    run_whisper(sys.argv[1], sys.argv[2])
