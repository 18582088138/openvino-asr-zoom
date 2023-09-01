import torch
import random
import os
from eval_ipex_cnn_true import Seq2Seq, extract_fbank, eval
from openvino.runtime import CompiledModel 

import openvino.runtime as ov
from openvino.tools import mo
from openvino.runtime import Core, serialize
import torch
import nncf, sys
from torch.utils.data import Dataset, TensorDataset, DataLoader

ie = Core()

# Define your custom dataset
class Seq2SeqDecoderDataset(Dataset):
    def __init__(self, decoder_seq_data, encoder_output_data, encoder_mask_data):
        self.decoder_seq_data = decoder_seq_data
        self.encoder_output_data = encoder_output_data
        self.encoder_mask_data = encoder_mask_data
    
    def __len__(self):
        return len(self.decoder_seq_data)
    
    def __getitem__(self, idx):
        return (
            self.decoder_seq_data[idx],
            self.encoder_output_data[idx],
            self.encoder_mask_data[idx]
        )
        
def print_model(model):
    model_size = sum(p.numel() for p in model.parameters()) / 1000000.
    print('Model size: %.2fM' % model_size)
    
def transform_fn(data_item):
    src = data_item
    return src
   
if __name__ == '__main__':
    
    # Set model_path before running
    model_path = "./models/with_new_eval_for_bm_comp/decoder/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)


    model_name = "zoom_dec_use_cnn"
    torch2ov = 1
    ov_nncf_quantize = 1
    
    if len(sys.argv) < 2:
        print("Please specify an audio file.")
        sys.exit()
    wav_file = sys.argv[1]
    
    use_cnn = True #False
    d_input = 40
    freq_kn=3
    freq_std=2
    
    print("Extract log mel features from: {}".format(wav_file))
    src = extract_fbank(wav_file)
    print("Audio length: {} ms ".format(src.size(0)*10))
    print_precision="FP32"
    
    model = Seq2Seq(enc_dropout=0.3,enc_dropconnect=0.3,n_vocab=4003, d_input=40, \
                    d_enc=1024, n_enc=6, d_dec=1024, n_dec=2,d_emb=512,d_project=300, \
                    n_head=1,use_cnn=True,dec_dropout=0.2,dec_dropconnect=0.2,emb_drop=0.15)

    model.eval()
    print('before quantize')
    print_model(model)
    
    src = extract_fbank(wav_file)

    src_shape = src.unsqueeze(0).shape
    pre_process_l = (((src_shape[1]-freq_kn)//freq_std + 1)-freq_kn)//freq_std + 1
    pre_process_d = ((((src_shape[2]-freq_kn)//freq_std + 1)-freq_kn)//freq_std + 1)*32
    
    input_dict = {
        "onnx::Unsqueeze_0" : src.unsqueeze(0),
        "onnx::Unsqueeze_1" : pre_process_l,
        "onnx::Unsqueeze_2" : pre_process_d,
    }

    torch_enc_model_path = model_path + model_name + ".pt"
    onnx_enc_model_path = model_path + model_name + ".onnx"
    ir_enc_model_path = model_path + model_name  + ".xml"

    torch_dec_model_path = model_path + model_name + ".pt"
    onnx_dec_model_path = model_path + model_name + ".onnx"
    ir_dec_model_path = model_path + model_name  + ".xml"
    
    if torch2ov:
        torch.save(model.decoder, torch_dec_model_path)
        enc_out = model.encode(src.unsqueeze(0), pre_process_l, pre_process_d, None)[0]
        enc_mask = torch.ones((1, enc_out.size(1)), dtype=torch.uint8)
        l = random.randint(5, 10)
        seq = torch.LongTensor([1] * 10)
        print(f"seq.shape = {seq.shape}")
        seq = seq.unsqueeze(0)
        
        model_inputs = (seq,  
                        enc_out, 
                        enc_mask)

        seq_dynamic_shape_ = {
            'input.1': {0: 'bs', 1: 'seq_len'},
            'onnx::MatMult_1': {0: 'bs', 1: 'fs'},
            'onnx::Cast_2': {0:'bs', 1: 'mask'}
        }
        
        input_names = ["input.1", "onnx::MatMult_1", "onnx::Cast_2"]
        torch.onnx.export(model.decoder,            
                        model_inputs,
                        onnx_dec_model_path,       # where to save the model (can be a file or file-like object)
                        input_names=input_names,
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=15,          # the ONNX version to export the model to
                        verbose = True,
                        dynamic_axes=seq_dynamic_shape_)
        import onnxruntime
        import numpy as np
        onnx_session = onnxruntime.InferenceSession(onnx_dec_model_path)
        
        inputs = {"input.1":seq.detach().numpy(), 
                  "onnx::MatMult_1":enc_out.detach().numpy(), 
                  "onnx::Cast_2":enc_mask.detach().numpy()}
        outputs = onnx_session.run(None, inputs)
        
        output_tensor = outputs[0]
        print(f"onnx runtime output: {outputs}")

        ov_model = mo.convert_model(onnx_dec_model_path)
        serialize(ov_model, str(ir_dec_model_path))
       
        ov_fp32_dec_model = ie.read_model(ir_dec_model_path)
        compiled_model_dec = ie.compile_model(ov_fp32_dec_model)
        outputs = compiled_model_dec(inputs)
        
    if ov_nncf_quantize:
        
        ov_model = ie.read_model(ir_dec_model_path)

        src = src.unsqueeze(0)
        enc_out = model.encode(src,pre_process_l, pre_process_d, None)[0]
        enc_mask = torch.ones((1, enc_out.size(1)), dtype=torch.uint8)

        
        l = random.randint(5, 10)
        dec_seq = torch.LongTensor([1] * 10)

        dec_seq = dec_seq.unsqueeze(0)
        
        enc_out = enc_out.detach().numpy()
        enc_mask = enc_mask.detach().numpy()
        dec_seq = dec_seq.detach().numpy()

        dataset = Seq2SeqDecoderDataset(dec_seq, enc_out, enc_mask)

        pot_dataloader = DataLoader(dataset)
        
        calibration_dataset = nncf.Dataset(pot_dataloader, transform_fn)
        quantized_model = nncf.quantize(ov_model, calibration_dataset)

        int8_ir_path = model_path + model_name + "_int8" + ".xml"
        ov.serialize(quantized_model, int8_ir_path)
        
        ov_int8_dec_model = ie.read_model(int8_ir_path)
        compiled_model_dec = ie.compile_model(ov_int8_dec_model)
        outputs = compiled_model_dec(inputs)
        
        
