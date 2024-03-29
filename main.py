import os
import time
import gdown
import json
import numpy as np

from scipy.io import wavfile
from scipy import signal
from degradation_utils import (prepare_audio, apply_degradation,
                              compose_degradations, save_audio)

# Whisper
from transformers import (WhisperForConditionalGeneration, 
                          WhisperFeatureExtractor, WhisperTokenizer, 
                          WhisperTokenizer, WhisperProcessor)

# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))

ROOT = os.path.dirname(os.path.realpath(__file__))

# Constants
RATE = 16e3 # Default rate of whisper

def load_finetuned(language="French"):
  """Download finetuned weights from drive"""
  if not os.path.isdir("./pretrained"):
    os.mkdir("./pretrained")
  st = time.time()
  # Load dictionary of ids
  with open(os.path.join(ROOT,"finetuned_models.json")) as file:
    dict_finetuned = json.load(file)  
  # Download 
  for f, file_id in dict_finetuned[language].items():
    gdown.download(f"https://drive.google.com/uc?id={file_id}&confirm=t",
                 output=f"./pretrained/{f}",
                 use_cookies=False)
  et = time.time()
  print(f"\nDownloaded finetuned weights in {et-st:.3f} seconds.")

def main(audio_in,  # audio files
          language, finetuned, force_language, # model 
          add_noise, snr, # degradation options
          impulse_response, wet_level,
          pitch_shift,
          time_stretch,
          dr_compression):
    
    f = open("transcription.txt", "a")
    f.close()
    
    ## Load audio
    sample_rate_in, samples = wavfile.read(audio_in)
    
    ## Preprocess audio
    # Stereo to Mono
    if len(samples.shape) == 2:
        samples = samples.mean(axis=1)
    # Normalize
    samples = prepare_audio(samples)        
    # Resample
    if sample_rate_in != RATE:
        samples, sample_rate = apply_degradation(["resample,16000"], 
                                                samples, sample_rate_in)
        assert sample_rate == RATE

    # Apply degradations
    list_degradations = compose_degradations(add_noise, snr,
                                            impulse_response, wet_level,
                                            pitch_shift,
                                            time_stretch,
                                            dr_compression)
    if len(list_degradations) !=0:
      samples, sample_rate = apply_degradation(list_degradations, 
                                                samples, RATE,
                                                verbose=1)
    # Trim audio
    max_len = int(RATE*30)
    if len(samples)>max_len:
      samples = samples[:max_len]
    # Write processed audio
    output_samples = samples / samples.max()
    save_audio(output_samples, RATE, save_file="output.wav")
    
    ## Load model
    if finetuned:
      try:
        load_finetuned(language)
      except Exception as e:
        print("\nFailed to download finetuned weights from Drive." 
              "Please refresh or try later.\n")
        print(e)
        exit()
      pretrained_path = "./pretrained"
    else:
      pretrained_path = "openai/whisper-tiny" 
      print(f"\nYou haven't checked finetuned weights. " 
            f"The pretrained model is the multilangual \"openai/whisper-tiny\".\n")


    ## Instanciate whisper pipeline
    # feature_extractor: 1. trim+pad 2. STFT 3. MEL cepstrum == features
    feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_path)
    # tokenizer
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny",
                                                language=language, 
                                                task="transcribe")
    # processor: wraps feature extractor + tokenizer 
    processor = WhisperProcessor(feature_extractor, tokenizer)
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_path)
    # Force task and language
    if force_language:
      forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, 
                                                          task="transcribe")
      model.generation_config.forced_decoder_ids = forced_decoder_ids
    ## Process audio with Whisper
    st = time.time()
    inputs = processor(samples, sampling_rate=RATE, return_tensors="pt")
    generated_ids = model.generate(inputs=inputs.input_features)
    transcription = processor.batch_decode(generated_ids, 
                                            skip_special_tokens=True)[0]
    et = time.time()
    print(f"\nTranscribed audio in {et-st:.3f} seconds.")
    
    f = open("transcription.txt", "w")
    f.write(transcription)
    f.close()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_in", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--force_language", type=str2bool, required=True)
    parser.add_argument("--finetuned", type=str2bool, required=True)
    parser.add_argument("--add_noise", type=str, required=True)
    parser.add_argument("--snr", type=float, required=True)
    parser.add_argument("--impulse_response", type=str, required=True)
    parser.add_argument("--wet_level", type=float, required=True)
    parser.add_argument("--pitch_shift", type=float, required=True)
    parser.add_argument("--time_stretch", type=float, required=True)
    parser.add_argument("--dr_compression", type=str, required=True)
    
    args = parser.parse_args()
    main(args.audio_in, 
          args.language, args.finetuned, args.force_language,
          args.add_noise, args.snr,
          args.impulse_response, args.wet_level,
          args.pitch_shift,
          args.time_stretch,
          args.dr_compression)
