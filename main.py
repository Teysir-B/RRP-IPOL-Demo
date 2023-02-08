import os
import iio
import numpy as np
from scipy.io import wavfile

# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))

def main(input, output, sigma):

    samplerate, data = wavfile.read(input)
    print(samplerate, data)
    wavfile.write(output, samplerate//2, data.astype(np.int16))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--language", type=float, required=True)
    #parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()
    print("Blablabla")
    #main(args.input, args.output, args.sigma)
