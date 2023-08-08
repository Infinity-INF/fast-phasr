import librosa
import torch
import whisper_ph_asr
#import genshin
def infer(model,inferdevice,file):
    if inferdevice=="自动":
        devices=torch.cuda.is_available()
        if devices==True:
            print("Use CUDA")
            device=torch.device('cuda')
            asr = whisper_ph_asr.PhonemeAsr().cuda()
        else:
            print("Use CPU")
            torch.device('cpu')
            asr = whisper_ph_asr.PhonemeAsr().cpu()
    else:
        devices = torch.cuda.is_available()
        print("Use CPU")
        torch.device('cpu')
        asr = whisper_ph_asr.PhonemeAsr().cpu()

    sounddir=file

    pth = model
    ckpt = torch.load(pth)
    asr.load_state_dict(ckpt)

    wav16k, _ = librosa.load(sounddir, sr=16000)
    phonemes, durations = whisper_ph_asr.get_asr_result(asr, wav16k)

    print(phonemes, durations)
    yield phonemes,durations
if __name__ =="__main__":
    infer()