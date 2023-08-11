import librosa
import torch
import whisper_ph_asr
import os

devices = torch.cuda.is_available()
if devices:
    print("Use CUDA")
    device = torch.device('cuda')
    asr = whisper_ph_asr.PhonemeAsr().cuda()
else:
    print("Use CPU")
    device = torch.device('cpu')
    asr = whisper_ph_asr.PhonemeAsr().cpu()

print("Input your sounds directory:")
sounddir = input()

pth = "phasr.pth"
ckpt = torch.load(pth)
asr.load_state_dict(ckpt)

wav16k, _ = librosa.load(sounddir, sr=16000)
phonemes, durations = whisper_ph_asr.get_asr_result(asr, wav16k)

htk_labels = []
current_time = 0

for phoneme, duration in zip(phonemes, durations):
    htk_label = f"{current_time} {current_time + int(duration * 10000000)} {phoneme}"
    htk_labels.append(htk_label)
    current_time += int(duration * 10000000)

output_filename = os.path.splitext(os.path.basename(sounddir))[0] + ".lab"
output_path = os.path.join(os.path.dirname(sounddir), output_filename)

with open(output_path, 'w') as f:
    for label in htk_labels:
        f.write(label + '\n')

print("HTK-style labels saved as:", output_path)
