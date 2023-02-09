import os
import iio
import gdown
import numpy as np

from scipy.io import wavfile
from scipy import signal

import whisper

# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))
RATE = 16e3
def main(audio_file, language, output_audio, noisy_audio, sigma):
    # Load audio
    datarate, audio = wavfile.read(audio_file)
    if len(audio.shape) == 2:
        # Stereo to Mono
        audio = audio.mean(axis=1)
    # Normalize audio
    audio = audio / audio.max() 
    audio = audio.astype(np.float16)
    # Resample
    if datarate != RATE:
        new_len = int(len(audio)*(RATE/datarate))
        audio = signal.resample(audio, new_len)
    audio = whisper.pad_or_trim(audio)
    wavfile.write(output_audio, datarate, audio)
    audio += np.random.normal(0, sigma, *audio.shape)
    wavfile.write(noisy_audio, datarate, audio)
    checkpoint_path = "./tiny_multilanguage.ckpt"
    gdown.download(url = "https://drive.google.com/uc?id=12v5I212oqXDvCsKnU5cSnbFt3eT28sax&confirm=t", 
                   output = checkpoint_path, quiet=False)
    
    model = whisper.load_model(checkpoint_path)
    #model = whisper.load_model("tiny")
    options = dict(language = language)
    transcribe_options = dict(task = "transcribe", **options)
    results = model.transcribe(audio, **transcribe_options)
    print(results["text"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--output_audio", type=str, required=True)
    parser.add_argument("--noisy_audio", type=str, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    
    args = parser.parse_args()
    main(args.audio_file, args.language, args.output_audio, args.noisy_audio, args.sigma)
