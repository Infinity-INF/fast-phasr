import librosa
import torch
import whisper_ph_asr

devices=torch.cuda.is_available()
if devices==True:
    print("Use CUDA")
    device=torch.device('cuda')
    asr = whisper_ph_asr.PhonemeAsr().cuda()
else:
    print("Use CPU")
    torch.device('cpu')
    asr = whisper_ph_asr.PhonemeAsr().cpu()

print("input your sounds directory:")
sounddir=input()

pth = "phasr.pth"
ckpt = torch.load(pth)
asr.load_state_dict(ckpt)

wav16k, _ = librosa.load(sounddir, sr=16000)
phonemes, durations = whisper_ph_asr.get_asr_result(asr, wav16k)

print(phonemes, durations)
