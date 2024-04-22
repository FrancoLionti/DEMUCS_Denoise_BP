import torch
import torchaudio
import os
from denoiser import pretrained
from denoiser.dsp import convert_audio

# Escribir el directorio donde se encuentran los archivos a denoisear
directory_input= r"Pretrained Denoisers\Input"

# Escribir el directorio donde se almacenaran los archivos denoiseados
directory_output= r"Pretrained Denoisers\Output"

input_dir = os.fsencode(directory_input)
output_dir = os.fsencode(directory_output)
model = pretrained.dns64().cpu()
for file in os.listdir(input_dir):
    filename = os.fsdecode(file)
    if filename.endswith(".mp3") or filename.endswith(".wav"): 
        # Modelo preentrenado
        wav, sr = torchaudio.load('Pretrained Denoisers\Input\\' + filename)
        wav = convert_audio(wav.cpu(), sr, model.sample_rate, model.chin)
        with torch.no_grad():
            denoised_tensor = model(wav[None])[0]
        torchaudio.save(directory_output + '\\' +'Denoised_' + filename,denoised_tensor, model.sample_rate)
        continue
    else:
        continue
