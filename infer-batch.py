import argparse
import os
import torch
import librosa
from tqdm import tqdm
import whisper_ph_asr

devices = torch.cuda.is_available()
if devices:
    print("Use CUDA")
    device = torch.device('cuda')
    asr = whisper_ph_asr.PhonemeAsr().cuda()
else:
    print("Use CPU")
    device = torch.device('cpu')
    asr = whisper_ph_asr.PhonemeAsr().cpu()

parser = argparse.ArgumentParser(description="Batch inference for audio files")
parser.add_argument("--batch", action="store_true", help="Enable batch inference mode")
parser.add_argument("input_dir", type=str, help="Input directory containing WAV files")
args = parser.parse_args()

def get_wav_file_list(input_dir):
    wav_file_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            wav_file_list.append(os.path.join(input_dir, filename))
    return wav_file_list

if args.batch:
    sounddir = args.input_dir
    wav_file_list = get_wav_file_list(sounddir)
else:
    print("Input your sounds directory:")
    sounddir = input()
    wav_file_list = [sounddir]

pth = "phasr.pth"
ckpt = torch.load(pth)
asr.load_state_dict(ckpt)

for wav_file in tqdm(wav_file_list, desc="Processing"):
    wav16k, _ = librosa.load(wav_file, sr=16000)
    phonemes, durations = whisper_ph_asr.get_asr_result(asr, wav16k)

    htk_labels = []
    current_time = 0

    for phoneme, duration in zip(phonemes, durations):
        htk_label = f"{current_time} {current_time + int(duration * 10000000)} {phoneme}"
        htk_labels.append(htk_label)
        current_time += int(duration * 10000000)

    output_filename = os.path.splitext(os.path.basename(wav_file))[0] + ".lab"
    output_path = os.path.join(os.path.dirname(wav_file), output_filename)

    with open(output_path, 'w') as f:
        for label in htk_labels:
            f.write(label + '\n')

    print("HTK-style labels saved as:", output_path)
    print("Inference completed for:", wav_file)
