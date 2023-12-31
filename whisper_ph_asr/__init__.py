import os

import librosa
import torch
from torch import nn

from . import commons
from . import attentions
from .whisper_encoder import AudioEncoder,  log_mel_spectrogram, pad_or_trim
ttsing_phone_set = ['_'] + [
    "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r",
    "s", "sh", "t", "x", "z", "zh", "a", "ai", "an", "ang", "ao", "e", "ei",
    "en", "eng", "er", "iii", "ii", "i", "ia", "ian", "iang", "iao", "ie", "in",
    "ing", "iong", "iou", "o", "ong", "ou", "u", "ua", "uai", "uan", "uang",
    "uei", "uen", "ueng", "uo", "v", "van", "ve", "vn", "AH", "AA", "AO", "ER",
    "IH", "IY", "UH", "UW", "EH", "AE", "AY", "EY", "OY", "AW", "OW", "P", "B",
    "T", "D", "K", "G", "M", "N", "NG", "L", "S", "Z", "Y", "TH", "DH", "SH",
    "ZH", "CH", "JH", "V", "W", "F", "R", "HH", "AH0", "AA0", "AO0", "ER0",
    "IH0", "IY0", "UH0", "UW0", "EH0", "AE0", "AY0", "EY0", "OY0", "AW0", "OW0",
    "AH1", "AA1", "AO1", "ER1", "IH1", "IY1", "UH1", "UW1", "EH1", "AE1", "AY1",
    "EY1", "OY1", "AW1", "OW1", "AH2", "AA2", "AO2", "ER2", "IH2", "IY2", "UH2",
    "UW2", "EH2", "AE2", "AY2", "EY2", "OY2", "AW2", "OW2", "AH3", "AA3", "AO3",
    "ER3", "IH3", "IY3", "UH3", "UW3", "EH3", "AE3", "AY3", "EY3", "OY3", "AW3",
    "OW3", "D-1", "T-1", "P*", "B*", "T*", "D*", "K*", "G*", "M*", "N*", "NG*",
    "L*", "S*", "Z*", "Y*", "TH*", "DH*", "SH*", "ZH*", "CH*", "JH*", "V*",
    "W*", "F*", "R*", "HH*", "sp", "sil", "or", "ar", "aor", "our", "angr",
    "eir", "engr", "air", "ianr", "iaor", "ir", "ingr", "ur", "iiir", "uar",
    "uangr", "uenr", "iir", "ongr", "uor", "ueir", "iar", "iangr", "inr",
    "iour", "vr", "uanr", "ruai", "TR", "rest",
    # opencpop
    'w', 'SP', 'AP', 'un', 'y', 'ui', 'iu',
    # opencpop-strict
    'i0', 'E', 'En',
    # japanese-common
    'ts.', 'f.', 'sh.', 'ry.', 'py.', 'h.', 'p.', 'N.', 'a.', 'm.', 'w.', 'ky.',
    'n.', 'd.', 'j.', 'cl.', 'ny.', 'z.', 'o.', 'y.', 't.', 'u.', 'r.', 'pau',
    'ch.', 'e.', 'b.', 'k.', 'g.', 's.', 'i.',
    # japanese-unique
    'gy.', 'my.', 'hy.', 'br', 'by.', 'v.', 'ty.', 'xx.', 'U.', 'I.', 'dy.'
]
ttsing_phone_to_int = {}
int_to_ttsing_phone = {}
for idx, item in enumerate(ttsing_phone_set):
    ttsing_phone_to_int[item] = idx
    int_to_ttsing_phone[idx] = item


LRELU_SLOPE = 0.1


hps = {
  "data": {
    "unit_dim": 768,
  },
  "model": {
    "hidden_channels": 192,
    "spk_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 4,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "prior_hidden_channels": 192,
    "prior_filter_channels": 768,
    "prior_n_heads": 2,
    "prior_n_layers": 4,
    "prior_kernel_size": 3,
    "prior_p_dropout": 0.1,
    "resblock": "1",
    "use_spectral_norm": False,
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "upsample_rates": [8,8,4,2],
    "upsample_initial_channel": 256,
    "upsample_kernel_sizes": [16,16,8,4],
    "n_harmonic": 64,
    "n_bands": 65
  }
}


class PhonemeAsr(nn.Module):
    """
    Model
    """

    def __init__(self):
        super().__init__()
        self.hps = hps

        self.pre_net = nn.Conv1d(768, hps["model"]["prior_hidden_channels"], 1)
        self.proj = nn.Conv1d(hps["model"]["prior_hidden_channels"], len(ttsing_phone_set), 1)
        self.encoder = attentions.Encoder(
            hps["model"]["prior_hidden_channels"],
            hps["model"]["prior_filter_channels"],
            hps["model"]["prior_n_heads"],
            hps["model"]["prior_n_layers"],
            hps["model"]["prior_kernel_size"],
            hps["model"]["prior_p_dropout"])
        self.whisper_model = AudioEncoder(80, 1500, 768, 12, 12)

    def forward(self, units):
        phone_lengths = torch.LongTensor([units.shape[2]]).to(units.device)
        x = self.pre_net(units)
        x_mask = torch.unsqueeze(commons.sequence_mask(phone_lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        x = self.proj(x)
        return x



def get_whisper_units(model=None, wav16k_numpy=None):
    dev = next(model.parameters()).device
    mel = log_mel_spectrogram(wav16k_numpy).to(dev)[:, :3000]
    # if torch.cuda.is_available():
    #     mel = mel.to(torch.float16)
    feature_len = mel.shape[-1] // 2
    assert  mel.shape[-1] < 3000, "输入音频过长，只允许输入30以内音频"
    with torch.no_grad():
        feature = model(pad_or_trim(mel, 3000).unsqueeze(0))[:1, :feature_len, :].cpu().transpose(1,2)
    return feature

def load_checkpoint(checkpoint_path, model):
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict)

def remove_consecutive_duplicates(lst):
    sr = 16000
    hop = 320
    new_lst = []
    dur_lst = []
    previous = None
    count = 1
    for item in lst:
        if item == previous:
            count += 1
        else:
            if previous:
                new_lst.append(f"{previous}")
                dur_lst.append(count*hop/sr)
            previous = item
            count = 1
    new_lst.append(f"{previous}")
    dur_lst.append(count*hop/sr)
    return new_lst, dur_lst

def convert_x_to_phones(x):
    phoneme_ids = torch.argmax(x, dim=1)
    phones, durs = remove_consecutive_duplicates([int_to_ttsing_phone[int(i)] for i in phoneme_ids[0, :]])
    return phones, durs

def load_phoneme_asr_model():
    # whisper_model = load_whisper_model()
    current_file = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file)
    checkpoint_path = f"{current_directory}/full_asr_model.pth"
    asr_model = PhonemeAsr(hps)
    _ = asr_model.eval()
    load_checkpoint(checkpoint_path, asr_model)
    if torch.cuda.is_available():
        asr_model = asr_model.cuda()
        # asr_model = asr_model.half()
    return asr_model

def get_asr_result(asr_model, wav16k_numpy):
    units = get_whisper_units(asr_model.whisper_model, wav16k_numpy)
    with torch.no_grad():
        if torch.cuda.is_available():
            units = units.cuda()
        x = asr_model(units)
        x = x.cpu()
    phones, durs = convert_x_to_phones(x)
    return phones, durs

def get_silent_result(asr_model, wav16k_numpy):
    units = get_whisper_units(asr_model.whisper_model, wav16k_numpy)
    with torch.no_grad():
        if torch.cuda.is_available():
            units = units.cuda()
        x = asr_model(units)
        x = x.cpu()
    phoneme_ids = torch.argmax(x, dim=1)
    phonemes = [int_to_ttsing_phone[int(i)] for i in phoneme_ids[0, :]]

    res_list = []
    previous = None
    for idx, item in enumerate(phonemes):
        if item != previous:
            if item in ["SP", "AP", "pau"]:
                res_list.append(item)
            else:
                res_list.append(None)

            previous = item
        else:
            res_list.append(None)


    # print(res_list)
    # print(len(phonemes))
    # print(sum([1 if i==j else 0 for i, j in zip(res_list, phonemes)]))
    return res_list

