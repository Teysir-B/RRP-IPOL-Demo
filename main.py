#import whisper
import os
import iio
import numpy as np
from scipy.io import wavfile
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.realpath(__file__))

def main(audio, language):
    print(audio)
    
    #model = whisper.load_model("base")
    # Set options
    #options = dict(language = language)
    #transcribe_options = dict(task = "transcribe", **options)
    #result = model.transcribe(audio, **transcribe_options)
    #print(result["text"])

if __name__ == "__main__":
    print("Blablabla")
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type = str, required=True)
    parser.add_argument("--language", type = str, required=True)

    args = parser.parse_args()
    
    main(args.audio, args.language)
