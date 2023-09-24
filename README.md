# fast-phasr
Phonemes labeling based on whisper small

## Use

### Install dependencies

```
# gpu
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# cpu
pip3 install torch torchvision torchaudio

pip install -r requirement.txt
```

### Inference

```
python infer.py [yor wav directory]
```

## Requirement

```
ffmpeg==1.4
librosa==0.10.0.post2
numpy==1.24.4
gradio
torch
```