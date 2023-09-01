import torch

from eval_ipex_cnn_true import Seq2Seq, extract_fbank, eval
from openvino.runtime import CompiledModel 

import openvino.runtime as ov
from openvino.tools import mo
from openvino.runtime import Core, serialize
import torch
from torch.utils.data import TensorDataset, DataLoader

ie = Core()

model_name = "zoom_encoder_rnn_use_cnn_no_ipex"
wav_file = "samples/common_voice_en_21108678.wav"
src = extract_fbank(wav_file)

use_cnn = True #False
d_input = 40
freq_kn=3
freq_std=2

model = Seq2Seq(n_vocab=4000, d_input=d_input, d_enc=1024, n_enc=6, d_dec=1024, n_dec=2, use_cnn=use_cnn)
src = src.unsqueeze(0)

if use_cnn:
    src = model.encoder.cnn(src.unsqueeze(1))
    src = src.permute(0, 2, 1, 3).contiguous()
    src = src.view(src.size(0), src.size(1), -1)
    
print(model.encoder.rnn)
torch.save(model.encoder.rnn.state_dict(), model_name + ".pt")

model.eval()
print(model.encoder.rnn)

torch.onnx.export(model.encoder.rnn,               # model being run
                  src,                         # model input (or a tuple for multiple inputs)
                  model_name + ".onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  verbose = True
                  )

ov_model = mo.convert_model(model_name + ".onnx", compress_to_fp16=True)

#ir_model = model_name + ".xml"
serialize(ov_model, str(model_name)+str(".xml"))

print(f"OV model (xml/bin) saved as:  {model_name}")