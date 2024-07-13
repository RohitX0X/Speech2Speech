

import whisper

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("./rec.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

#model = AutoModelForSeq2SeqLM.from_pretrained("C:/Users/rushi/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d")


#article = "Hello Rohit, Hope i am audible , i want to tell i am able to hear you and i miss you  "
inputs = tokenizer(result.text, return_tensors="pt")

translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva"), max_length=50)

s = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

print(s)
# from transformers import pipeline
import scipy
# import numpy as np
# synthesiser = pipeline("text-to-speech", "suno/bark")
import torch
import torchaudio



from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav



import torch
from transformers import VitsTokenizer, VitsModel, set_seed

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-hin")
model = VitsModel.from_pretrained("facebook/mms-tts-hin")

inputs = tokenizer(text=s, return_tensors="pt")

set_seed(555)  # make deterministic

with torch.no_grad():
   outputs = model(**inputs)

waveform = outputs.waveform[0]

print(waveform.numpy())
#torchaudio.save("output.wav", torch.from_numpy(waveform.numpy()), 24000)

scipy.io.wavfile.write("bark_out.wav", 12000, data = waveform.numpy())
