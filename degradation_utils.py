import time
from typing import List
import numpy as np
import audio_degrader as ad

def compose_degradations(add_noise, snr, impulse_response, wet_level):
  """ Compose degradation json file from parameters """
  list_degradations = []
  # Additive noise
  if add_noise != "none":
    list_degradations.append(f"mix,{add_noise},{snr}")
  # Convolution + Reverberation
  if impulse_response != "none":
    list_degradations.append(f"convolution,{impulse_response},{wet_level}")
  # ...
  if len(list_degradations) !=0 :
    print("\nDegradation Composition:")
    for d in list_degradations:
      print(f"\t{d}")
  
  return list_degradations

def prepare_audio(samples):
    # Normalize (+other if needed)
    if np.abs(samples).max() > 1:
      samples = samples.astype(np.float32)
      rms_samples = np.sqrt(np.sum(np.power(samples, 2)))
      samples = samples/rms_samples
    return samples

def apply_degradation(degradation: List[str], samples, 
                      sample_rate_in: int = 16e3, verbose=0):
  """
  Function to apply degradations on one audio samples.
  Inputs:
  - degradation: List of strings. String format name_degradation,param1,param2.
  - samples: Numpy array of samples.
  - sample_rate_in: Sample rate of the audio. 
                    If not equal to 16k Hz, will be resampled.
  """
  st = time.time()
  # Prep audio
  audio = ad.AudioArray(samples_in = samples, 
                      sample_rate_in = sample_rate_in,
                      sample_rate_process= 16e3, # Default rate
                      bits = 64) # float16 for Whisper
  # Loop over degradations and apply
  degradation = ad.ParametersParser.parse_degradations_args(degradation)
  for d in degradation:
    audio.apply_degradation(d)
  et = time.time()
  if verbose>0:
    print(f"\nApplied degradations in {et-st:.3f} seconds.")
  return audio.samples, int(audio.sample_rate)

