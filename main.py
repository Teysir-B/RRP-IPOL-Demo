import os
import iio
import numpy as np

from scipy.io import wavfile
from scipy import signal

import whisper

# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))

def main(input, output, sigma):

    rate = 16e3
    # Load audio
    datarate, audio = wavfile.read(input)
    # Normalize audio
    audio = audio / audio.max() 
    audio = audio.astype(np.float16)
    # Resample
    if datarate != rate:
        new_len = int(len(audio)*(rate/datarate))
        audio = signal.resample(audio, new_len)
    audio = whisper.pad_or_trim(audio)
    model = whisper.load_model("base.en")
    results = model.transcribe(audio, task = "transcribe", verbose = 1, fp16 = False)
    print(results["text"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()
    main(args.input, args.output, args.sigma)
