import time
from typing import List
import numpy as np
import sox
import audio_degrader as ad


def compose_degradations(add_noise="none", snr=None, 
                         impulse_response="none", wet_level=None, 
                          pitch_shift=1, time_stretch=1, 
                          dr_compression="none"):
  """ Compose degradation json file from parameters """
  list_degradations = []
  # Pitch Shift
  if pitch_shift !=1:
    list_degradations.append(f"pitch_shift,{pitch_shift}")
  # Time stretch
  if time_stretch !=1:
    list_degradations.append(f"time_stretch,{time_stretch}")
  # Dynamic range comression
  if dr_compression != "none":
    dr_compression = int(dr_compression)
    list_degradations.append(f"dr_compression,{dr_compression}")
  # Convolution + Reverberation
  if impulse_response != "none":
    list_degradations.append(f"convolution,{impulse_response},{wet_level}")
  # Additive noise
  if add_noise != "none":
    list_degradations.append(f"mix,{add_noise},{snr}")
  
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
  
def save_audio(samples, sample_rate, save_file):
    tfm = sox.Transformer()
    tfm.set_output_format(rate=sample_rate, 
                          bits=32, channels=1)
    tfm.build_file(input_array=samples, 
                    sample_rate_in=sample_rate,
                    output_filepath='output.wav'
    )
    
def apply_degradation(degradation: List[str], samples, 
                      sample_rate_in: int = 16e3,
                      save_file = None, 
                      verbose=0):
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

  if save_file is not None:
    tfm = sox.Transformer()
    tfm.set_output_format(rate=audio.sample_rate, 
                          bits=32, channels=2)
    tfm.build_file(input_array=audio.samples, 
                    sample_rate_in=audio.sample_rate,
                    output_filepath='output.wav'
    )
  if verbose>0:
    print(f"\nApplied degradations in {et-st:.3f} seconds.")
  return audio.samples, int(audio.sample_rate)
