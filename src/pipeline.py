from pathlib import Path
import subprocess


class SpeechPipeline:
    def __init__(self,
                 solospeech_script: str = "src/SoloSpeech/scripts/test_v2.py",
                 whisper_script: str = "src/whisper_runner.py"):
        """Simple speech pipeline wrapper."""
        root = Path(__file__).resolve().parent.parent
        self.solospeech_script = (root / solospeech_script).resolve()
        self.whisper_script = (root / whisper_script).resolve()

    def inference(self, input_wav: str, enroll_wav: str,
                  output_path: str, return_text: bool = False) -> str:
        """Run pipeline on one input audio."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        solospeech_out = output_path.with_suffix(".wav")

        # Step 1: SoloSpeech
        print(f"[INFO] Running SoloSpeech on {input_wav}")
        subprocess.run([
            "python3", str(self.solospeech_script),
            "--test-wav", str(input_wav),
            "--enroll-wav", str(enroll_wav),
            "--output-path", str(solospeech_out)
        ], check=True)

        # Step 2: Whisper
        print(f"[INFO] Running Whisper on {solospeech_out}")
        subprocess.run([
            "python3", str(self.whisper_script),
            str(solospeech_out),
            str(output_path)
        ], check=True)

        print(f"[DONE] Transcript saved at {output_path}")

        return (output_path.read_text(encoding="utf-8")
                if return_text else str(output_path))
