import argparse
import os
import torch
import librosa
import sys
from tqdm import tqdm
import yaml

import whisper_ph_asr

with open('phoneme_combinations_to_pinyin_dict.yaml', 'r', encoding='utf-8') as dict_file:
    yaml_dict = yaml.safe_load(dict_file)
    phoneme_combinations_to_pinyin_dict = {}
    for entry in yaml_dict['entries']:
        grapheme = entry['grapheme']
        phonemes = ''.join(entry['phonemes'])
        phoneme_combinations_to_pinyin_dict[grapheme] = phonemes

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
parser.add_argument("input_dir", type=str, help="Input directory containing WAV files")
args = parser.parse_args()

def get_wav_file_list(input_dir):
    wav_file_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            wav_file_list.append(os.path.join(input_dir, filename))
    return wav_file_list

sounddir = args.input_dir
wav_file_list = get_wav_file_list(sounddir)

pth = "phasr.pth"
ckpt = torch.load(pth)
asr.load_state_dict(ckpt)

for wav_file in tqdm(wav_file_list, desc="Processing", ncols=100):
    wav16k, _ = librosa.load(wav_file, sr=16000)
    phonemes, durations = whisper_ph_asr.get_asr_result(asr, wav16k)

    clean_phonemes = [p for p in phonemes if p not in {"SP", "AP"}]
    pinyin_sequence = ""
    i = 0
    while i < len(clean_phonemes):
        combined_pinyin = None
        for j in range(len(clean_phonemes), i, -1):
            phoneme_combination = ''.join(clean_phonemes[i:j])
            if phoneme_combination in phoneme_combinations_to_pinyin_dict:
                combined_pinyin = phoneme_combinations_to_pinyin_dict[phoneme_combination]
                break
        if combined_pinyin:
            pinyin_sequence += combined_pinyin + " "
            i = j
        else:
            pinyin_sequence += phoneme_combinations_to_pinyin_dict.get(clean_phonemes[i], clean_phonemes[i]) + " "
            i += 1

    output_filename = os.path.splitext(os.path.basename(wav_file))[0] + ".lab"
    output_path = os.path.join(os.path.dirname(wav_file), output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pinyin_sequence.rstrip())

    sys.stdout.write("\rProcessing: {:3}%|".format(int((wav_file_list.index(wav_file) + 1) / len(wav_file_list) * 100)))
    sys.stdout.flush()

print("\nInference completed")
