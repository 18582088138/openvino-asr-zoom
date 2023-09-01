import pandas as pd
import torch
import time, sys, os

from datasets import load_dataset
from eval_ipex_cnn_true import Seq2Seq, extract_fbank

from openvino.runtime import CompiledModel 
import openvino.runtime as ov
from openvino.runtime import Core, serialize

ie = Core()

def summerize_runtime(runtime_dict):
    df = pd.DataFrame(runtime_dict)
    print(df.sum())
    print(df.mean())
    print(df.std())

def ov_eval(model_enc, model_dec, input_tensors, warmup_iter=10):

    # initialize the hid = (h0, c0) here, because traced model needs it 
    enc_h0 = torch.zeros(size=(12, 1, 1024), dtype = torch.float32)
    enc_c0 = torch.zeros(size=(12, 1, 1024), dtype = torch.float32)
    enc_state = (enc_h0, enc_c0)

    dec_h0 = torch.zeros(size=(2, 1, 1024), dtype = torch.float32)
    dec_c0 = torch.zeros(size=(2, 1, 1024), dtype = torch.float32)
    dec_state = (dec_h0, dec_c0)

    runtime_dict = {"enc_time":[], "dec_time":[]}
    cnt = 0
    
    compiled_model_enc = ie.compile_model(model_enc)
    #enc_infer_request = compiled_model_enc.create_infer_request()
    
    compiled_model_dec = ie.compile_model(model_dec)
    #dec_infer_request = compiled_model_dec.create_infer_request()
    wav_file = "./samples/common_voice_en_21108678.wav"
    # wav_file = "./samples/hf_audio/wave/1272-128104-0000.wav"
    
    for i in range (3):
        for src in input_tensors:
            cnt += 1
            stime = time.time()
            
            src = extract_fbank(wav_file).unsqueeze(0)

            src_shape = src.shape
            pre_process_l = (((src_shape[1]-freq_kn)//freq_std + 1)-freq_kn)//freq_std + 1
            pre_process_d = ((((src_shape[2]-freq_kn)//freq_std + 1)-freq_kn)//freq_std + 1)*32
            input_dict = {
                "onnx::Unsqueeze_0" : src,
                "onnx::Unsqueeze_1" : pre_process_l,
                "onnx::Unsqueeze_2" : pre_process_d,
            }

            enc_out = compiled_model_enc(input_dict)
            # enc_out = compiled_model_enc(src, enc_state)
            
            enc_time = 1000*(time.time() - stime)
            if cnt > warmup_iter:
                runtime_dict["enc_time"].append(enc_time)

            stime = time.time()
            bs, w, h = enc_out[0].shape
            enc_mask = torch.ones((1, w), dtype=torch.uint8)

            for i in range(20):
                l = i + 1
                seq = torch.LongTensor([l] * l)

                dec_out = compiled_model_dec({"input.1":seq.unsqueeze(0), 
                  "onnx::MatMult_1":enc_out[0],
                  "onnx::Cast_2":enc_mask})
                
            decode_time = 1000*(time.time() - stime)

            if cnt > warmup_iter:
                runtime_dict["dec_time"].append(decode_time)
    return runtime_dict


def ov_quantized_model(model_int8_enc, model_int8_dec, input_tensors):
    print('run ov qunatized decoder and encoder')
    
    runtime_dict = ov_eval(model_int8_enc, \
                               model_int8_dec, \
                               input_tensors)
    summerize_runtime(runtime_dict)
    
    
if __name__ == "__main__":

    # ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # input_tensors = []
    # for idx in range(len(ds)):
    #     sample = ds[idx]["audio"]
    #     path = sample["path"]
    #     src = extract_fbank(sample["path"]).unsqueeze(0)
    #     input_tensors.append(src)

    input_tensors = []
    wav_files = os.listdir("./samples/wave")
    for wav_file in wav_files:
        src = extract_fbank("./samples/wave/" + str(wav_file)).unsqueeze(0)
        input_tensors.append(src)
        
    ### Instantiate the model
    use_cnn = True #False
    d_input = 40
    freq_kn=3
    freq_std=2

    model = Seq2Seq(n_vocab=4000, d_input=d_input, d_enc=1024, n_enc=6, d_dec=1024, n_dec=2, use_cnn=use_cnn)
    model.eval()

    # Set the paths before running the script    
    # ov_int8_enc_model = ie.read_model("./models/with_new_eval_for_bm_comp/encoder/zoom_full_enc_use_cnn.xml")
    # ov_int8_dec_model = ie.read_model("./models/with_new_eval_for_bm_comp/decoder/zoom_dec_use_cnn.xml")
    ov_int8_enc_model = ie.read_model("./models/with_new_eval_for_bm_comp/encoder/zoom_full_enc_use_cnn_int8.xml")
    ov_int8_dec_model = ie.read_model("./models/with_new_eval_for_bm_comp/decoder/zoom_dec_use_cnn_int8.xml")
    
    ov_quantized_model(ov_int8_enc_model, ov_int8_dec_model, input_tensors)