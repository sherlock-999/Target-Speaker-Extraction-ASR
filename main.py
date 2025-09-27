import argparse
from src.pipeline import SpeechPipeline


def main():
    parser = argparse.ArgumentParser(description="Run speech pipeline")
    parser.add_argument("--input", required=True, help="Path to input wav file")
    parser.add_argument("--enroll", required=True, help="Path to enrollment wav file")
    parser.add_argument("--output", required=True, help="Path for transcript txt")
    parser.add_argument("--return-text", action="store_true")
    args = parser.parse_args()

    pipeline = SpeechPipeline()
    result = pipeline.inference(args.input, args.enroll, args.output, args.return_text)

    if args.return_text:
        print("\n[TRANSCRIPT]:\n")
        print(result)


if __name__ == "__main__":
    main()
