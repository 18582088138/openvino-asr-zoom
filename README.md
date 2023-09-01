### OpenVINO Model optimization and evaluation pipeline for Automatic Speech Recognition (ASR) Seq2Seq model

## Pre-requisites
1. [Mini Conda](https://docs.conda.io/en/latest/miniconda.html)
2. Ubuntu 20.04, Ubuntu 22.04 or RedHat 8.7 host

### Environment setup 

Create Conda environment

```
conda create -n zov python==3.11
conda activate zov
```

Install below python packages
```
pip install torch==2.0.0
pip install torchaudio==2.0.0
pip install torchvision==0.14.0
pip install openvino==2023.1.0.dev20230728
pip install onnx==1.14.0
pip install onnxruntime==1.15.1
pip install nncf==2.5.0
pip install librosa==0.10.0.post2
pip install hf==0.0.4
pip install huggingface-hub==0.16.4
pip install datasets==2.14.4
```


#### Generate IR (FP32 and INT8) for encoder module
```python convert_to_mo_for_encoder.py ./samples/wave/1272-128104-0000.wav```

#### Generate IR (FP32 and INT8) for decoder module
```python convert_to_mo_for_decoder.py ./samples/wave/common_voice_en_21108678.wav```

**Note:** Update model_paths and samples directory path in above encoder and decoder model gerantion scripts

#### Running evlaution script with hf_audio samples
```python evaluate_ov_quantized.py```

**Note: ** Encoder module currently supports fixed size seq length (for input 1272-128104-0000.wav file).