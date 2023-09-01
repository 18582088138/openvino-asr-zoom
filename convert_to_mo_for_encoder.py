import os
import torch
import numpy as np

from eval_ipex_cnn_true import Seq2Seq, extract_fbank, eval
from openvino.runtime import CompiledModel 

import openvino.runtime as ov
from openvino.tools import mo
from openvino.runtime import Core, serialize
import nncf, sys
from torch.utils.data import Dataset, TensorDataset, DataLoader

ie = Core()

def print_model(model):
    model_size = sum(p.numel() for p in model.parameters()) / 1000000.
    print('Model size: %.2fM' % model_size)
    
def transform_fn(data_item):
    src = data_item[0]
    src_shape = src.shape
    pre_process_l = (((src_shape[1]-freq_kn)//freq_std + 1)-freq_kn)//freq_std + 1
    pre_process_d = ((((src_shape[2]-freq_kn)//freq_std + 1)-freq_kn)//freq_std + 1)*32
    return src, pre_process_l, pre_process_d
   
if __name__ == '__main__':
    
    #model_path = "set/location/to/save/fp32/model"
    model_path = "./models/with_new_eval_for_bm_comp/encoder/"
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    model_name = "zoom_full_enc_use_cnn"
    torch2ov = 1
    ov_nncf_quantize = 1
    
    if len(sys.argv) < 2:
        print("Please specify an audio file.")
        sys.exit()
    wav_file = sys.argv[1]
    # wav_file = "./samples/wave/1272-128104-0000.wav"
    
    use_cnn = True
    d_input = 40
    freq_kn=3
    freq_std=2
    conv2d_shape = 32
    
    print("Extract log mel features from: {}".format(wav_file))
    src = extract_fbank(wav_file)
    print("Audio length: {} ms ".format(src.size(0)*10))
    print_precision="FP32"

    src = extract_fbank(wav_file)
    print(f"src shape == {src.unsqueeze(0).shape}")

    src_shape = src.unsqueeze(0).shape
    pre_process_l = (((src_shape[1]-freq_kn)//freq_std + 1)-freq_kn)//freq_std + 1
    pre_process_d = ((((src_shape[2]-freq_kn)//freq_std + 1)-freq_kn)//freq_std + 1)*32
    d_input = pre_process_d
    
    input_dict = {
        "onnx::Unsqueeze_0" : src.unsqueeze(0),
        "onnx::Unsqueeze_1" : pre_process_l,
        "onnx::Unsqueeze_2" : pre_process_d,
    }
    
    model = Seq2Seq(enc_dropout=0.3,enc_dropconnect=0.3,n_vocab=4003, d_input=d_input, d_enc=1024, \
                    n_enc=6, d_dec=1024, n_dec=2,d_emb=512,d_project=300,n_head=1, \
                    use_cnn=True,dec_dropout=0.2,dec_dropconnect=0.2,emb_drop=0.15)

    model.eval()
    print('before quantize')
    print_model(model)

    torch_enc_model_path = model_path + model_name + ".pt"
    onnx_enc_model_path = model_path + model_name + ".onnx"
    ir_enc_model_path = model_path + model_name  + ".xml"
    
    if torch2ov:
        torch.save(model.encoder, torch_enc_model_path)
        
        enc_dynamic_shape_ = {
           'onnx::Unsqueeze_0': {0: 'bs', 1: 'ts', 2:'fs'},
        }
        
        torch.onnx.export(model.encoder,          # model being run
                        (src.unsqueeze(0), pre_process_l, pre_process_d),  # model input (or a tuple for multiple inputs)
                        onnx_enc_model_path,      # where to save the model (can be a file or file-like object)
                        input_names=["onnx::Unsqueeze_0","onnx::Unsqueeze_1","onnx::Unsqueeze_2"],
                        export_params=True,       # store the trained parameter weights inside the model file
                        opset_version=15,         # the ONNX version to export the model to
                        dynamic_axes=enc_dynamic_shape_,
                        verbose = True,
                        )
        
        # #### Look into compress_to_fp16 flag (for CPU)
        ov_model = mo.convert_model(onnx_enc_model_path) #, compress_to_fp16=True)
        serialize(ov_model, str(ir_enc_model_path))

        input_dict = {
            "onnx::Unsqueeze_0":src.unsqueeze(0),
            "onnx::Unsqueeze_1" : pre_process_l,
            "onnx::Unsqueeze_2" : pre_process_d
        }

        # Sample inference    
        ov_fp32_enc_model = ie.read_model(ir_enc_model_path)
        compiled_model_enc = ie.compile_model(ov_fp32_enc_model)
        enc_infer_req = compiled_model_enc.create_infer_request()
        # outputs = enc_infer_req.infer(input_dict)
        outputs = compiled_model_enc(input_dict)
        
    if ov_nncf_quantize:
        
        enc_model = ie.read_model(ir_enc_model_path)

        src = src.unsqueeze(0)
        src = src.detach().numpy()
        pot_dataset = TensorDataset(torch.from_numpy(src))
        pot_dataloader = DataLoader(pot_dataset)
       
        calibration_dataset = nncf.Dataset(pot_dataloader, transform_fn)
        
        quantized_model = nncf.quantize(enc_model, calibration_dataset)
         
        int8_ir_path = model_path + model_name + "_int8" + ".xml"
        ov.serialize(quantized_model, int8_ir_path)
