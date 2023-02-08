#import whisper
from scipy.io import wavfile

def main(audio, language):
    print(audio)
    
    #model = whisper.load_model("base")
    # Set options
    #options = dict(language = language)
    #transcribe_options = dict(task = "transcribe", **options)
    #result = model.transcribe(audio, **transcribe_options)
    #print(result["text"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type = str, required=True)
    parser.add_argument("--language", type = str, required=True)

    args = parser.parse_args()
    print("Blablabla")
    main(args.audio, args.language)
