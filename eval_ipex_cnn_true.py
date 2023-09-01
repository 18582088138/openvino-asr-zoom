import random

import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#import intel_extension_for_pytorch as ipex
from statistics import stdev

import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch import Tensor

import torch
import torch.nn as nn
from torch.jit import ScriptModule, script_method
from typing import Optional, Tuple

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout=0.1, residual=False):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.norm = nn.LayerNorm(n_feat)
        self.w_q = nn.Linear(n_feat, n_feat, bias=False)
        self.w_k = nn.Linear(n_feat, n_feat, bias=False)
        self.w_v = nn.Linear(n_feat, n_feat, bias=False)
        self.w_out = nn.Linear(n_feat, n_feat, bias=False)
        self.drop = nn.Dropout(dropout)
        self.residual = residual

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.w_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.w_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.w_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, scores.size(1), -1, -1)
            scores = scores.masked_fill(mask, -np.inf)
        attn =  torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        attn = self.drop(attn)
        x = torch.matmul(attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous()
        x = x.view(n_batch, -1, self.h*self.d_k) # (batch, time1, d_model)
        x = self.w_out(x) # (batch, time1, d_model)

        return x, attn

    def forward(self, query, key=None, mask=None, scale=1.):
        residual = query if self.residual else None
        query = self.norm(query)
        if key is None: key = query
        value = key

        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        x, attn = self.forward_attention(v, scores, mask)
        x = self.drop(x) * scale
        x = x if residual is None else (x + residual)
        return x, attn

class WrappedLSTM(nn.Module):
    def __init__(self, d_input, d_model, n_layer, batch_first, bidirectional, bias, dropout):
        super(WrappedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=d_input, hidden_size=d_model, num_layers=n_layer, batch_first=batch_first,
                        bidirectional=bidirectional, bias=bias, dropout=dropout)

    def forward(self, input, hx=None):
        return self.lstm(input, hx)

class Encoder(nn.Module):
    def __init__(self, d_input, d_model, n_layer, unidirect=False,
                 dropout=0.2, dropconnect=0., time_ds=1, use_cnn=False, freq_kn=3, freq_std=2, pack=False):
        super().__init__()

        self.time_ds = time_ds

        if use_cnn:
            cnn = [nn.Conv2d(1, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)), nn.ReLU(),
                   nn.Conv2d(32, 32, kernel_size=(3, freq_kn), stride=(2, freq_std)), nn.ReLU()]
            self.cnn = nn.Sequential(*cnn)
            # d_input = ((((d_input - freq_kn) // freq_std + 1) - freq_kn) // freq_std + 1) * 32
        else:
            self.cnn = None

        # self.rnn = nn.LSTM(input_size=d_input, hidden_size=d_model, num_layers=n_layer, batch_first=True,
        #                 bidirectional=(not unidirect), bias=True, dropout=dropout)
        # The current IPEX requires nn.LSTM to be in a module so that it could be quantized
        self.rnn = WrappedLSTM(d_input=d_input, d_model=d_model, n_layer=n_layer, batch_first=True, bidirectional=(not unidirect), bias=True, dropout=dropout)
        self.unidirect = unidirect
        self.pack = pack
    
    def rnn_fwd(self, seq,  hid=None):
        seq, hid = self.rnn(seq, hid)
        return seq, hid

    def forward(self, seq, seq_shape_l, seq_shape_d,  hid= None):
        if self.time_ds > 1:
            ds = self.time_ds
            l = ((seq.size(1) - 3) // ds) * ds
            seq = seq[:, :l, :]
            seq = seq.view(seq.size(0), -1, seq.size(2) * ds)
            # seq = seq.view(seq.size(0), -1, seq.size(2) * ds)

        if self.cnn is not None:
            seq = self.cnn(seq.unsqueeze(1))
            seq = seq.permute(0, 2, 1, 3).contiguous()
            seq = seq.view(seq.size(0), seq_shape_l, seq_shape_d)
            # seq = seq.view(seq.size(0), seq.size(1), -1)

        seq, hid = self.rnn_fwd(seq, hid)

        if not self.unidirect:
            hidden_size = seq.size(2) // 2
            seq = seq[:, :, :hidden_size] + seq[:, :, hidden_size:]

        return seq, hid

class Decoder(nn.Module):
    def __init__(self, n_vocab, d_model, n_layer, d_enc, d_emb=0, d_project=0,
            n_head=8, shared_emb=True, dropout=0.2, dropconnect=0., emb_drop=0., pack=True):
        super().__init__()

        # Define layers
        d_emb = d_model if d_emb==0 else d_emb
        self.emb = nn.Embedding(n_vocab, d_emb, padding_idx=0)
        self.emb_drop = nn.Dropout(emb_drop)
        self.scale = d_emb**0.5

        self.attn = MultiHeadedAttention(n_head, d_model, dropout, residual=False)
        dropout = (0 if n_layer == 1 else dropout)
        self.lstm = nn.LSTM(d_emb, d_model, n_layer, batch_first=True, dropout=dropout)
        self.transform = None if d_model==d_enc else nn.Linear(d_enc, d_model, bias=False)
        d_project = d_model if d_project==0 else d_project
        self.project = None if d_project==d_model else nn.Linear(d_model, d_project, bias=False)
        self.output = nn.Linear(d_project, n_vocab, bias=False)
        #nn.init.xavier_normal_(self.project.weight)
        if shared_emb: self.emb.weight = self.output.weight
        self.pack = pack

    def forward(self, dec_seq, enc_out, enc_mask, hid=None):
        dec_emb = self.emb(dec_seq) * self.scale
        dec_emb = self.emb_drop(dec_emb)

        if self.pack and dec_seq.size(0) > 1 and dec_seq.size(1) > 1:
            lengths = dec_seq.gt(0).sum(-1); #lengths[0] = dec_seq.size(1)
            dec_in = pack_padded_sequence(dec_emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            dec_out, hid = self.lstm(dec_in, hid)
            dec_out = pad_packed_sequence(dec_out, batch_first=True)[0]
        else:
            dec_out, hid = self.lstm(dec_emb, hid)
        
        enc_out = self.transform(enc_out) if self.transform is not None else enc_out
        lt = dec_out.size(1)
        attn_mask = enc_mask.eq(0).unsqueeze(1).expand(-1, lt, -1)

        context, attn = self.attn(dec_out, enc_out, mask=attn_mask)
        out = context + dec_out
        out = self.project(out) if self.project is not None else out
        out = self.output(out)

        return out, attn, hid

class Seq2Seq(nn.Module):
    def __init__(self, n_vocab=1000, d_input=40, d_enc=320, n_enc=6, d_dec=320, n_dec=2,
            unidirect=False, d_emb=0, d_project=0, n_head=8, shared_emb=False,
            time_ds=1, use_cnn=False, freq_kn=3, freq_std=2, enc_dropout=0.2, enc_dropconnect=0.,
            dec_dropout=0.1, dec_dropconnect=0., emb_drop=0., pack=True):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(d_input, d_enc, n_enc,
                            unidirect=unidirect, time_ds=time_ds,
                            use_cnn=use_cnn, freq_kn=freq_kn, freq_std=freq_std,
                            dropout=enc_dropout, dropconnect=enc_dropconnect, pack=pack)

        self.decoder = Decoder(n_vocab, d_dec, n_dec, d_enc,
                            n_head=n_head, d_emb=d_emb, d_project=d_project, shared_emb=shared_emb,
                            dropout=dec_dropout, dropconnect=dec_dropconnect, emb_drop=emb_drop, pack=pack)

    def forward(self, src_seq, src_mask, tgt_seq, encoding=True):
        if encoding:
            enc_out, enc_mask = self.encoder(src_seq, src_mask)[0:2]
        else:
            enc_out, enc_mask = src_seq, src_mask

        dec_out = self.decoder(tgt_seq, enc_out, enc_mask)[0]
        return dec_out, enc_out, enc_mask

    # def encode(self, src_seq, hid=None):
    #     return self.encoder(src_seq, hid)    
    def encode(self, src_seq, seq_shape_l,seq_shape_d, hid=None):
        return self.encoder(src_seq, seq_shape_l, seq_shape_d, hid)

    def decode(self, enc_out, enc_mask, tgt_seq, state=None):
        # if state is not None:
        #     hid, cell = zip(*state)
        #     state = (torch.stack(hid, dim=1), torch.stack(cell, dim=1))
        # print(len(state))
        logit, attn, state = self.decoder(tgt_seq, enc_out, enc_mask, state)
        hid, cell = state
        state = [(hid[:,j,:], cell[:,j,:]) for j in range(logit.size(0))]
        logit = logit[:,-1,:].squeeze(1)
        return torch.log_softmax(logit, -1), state, attn


def print_model(model):
    model_size = sum(p.numel() for p in model.parameters()) / 1000000.
    print('Model size: %.2fM' % model_size)


def extract_fbank(wav_file, sample_rate=16000, nfft=256,
        filters=40, frame_length=25., frame_shift=10., wav_dither=0.):

    signal = torchaudio.load(wav_file)[0]
    mat = kaldi.fbank(signal,
        num_mel_bins=filters,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=wav_dither,
        sample_frequency=sample_rate)

    return mat.detach()
def eval(model,src):
    model.eval()
    # print(model)
    total_encode, total_decode = [],[]
    print("Starting encoding and decoding with the Seq2Seq model...")
    for i in range(20):
        encode_time, decode_time = eval_(model,src)
        total_encode.append(encode_time)
        total_decode.append(decode_time)
    avg_encode = sum(total_encode)/len(total_encode)
    avg_decode = sum(total_decode)/len(total_decode)

    print("Avg Encoding: {}".format(avg_encode))
    print("Encoding STD: {}".format(stdev(total_encode)))

    print("Avg Decoding: {}".format(avg_decode))
    print("Decoding STD: {}".format(stdev(total_decode)))


    print("Done.")

def eval_(model,src):
    stime = time.perf_counter()

    # initialize the hid = (h0, c0) here, because traced model needs it 
    h0 = torch.zeros(size=(12,1,1024), dtype = src.dtype)
    hid = (h0, h0)
    enc_out = model.encode(src.unsqueeze(0), hid)[0]
    encode_time = time.perf_counter() - stime
    stime = time.perf_counter()
    enc_mask = torch.ones((1, enc_out.size(1)), dtype=torch.uint8)
    for i in range(10):
        l = i + 1
        seq = torch.LongTensor([l] * l)
        #print(seq.size())
        dec_out = model.decode(enc_out, enc_mask, seq.unsqueeze(0))[0]
        # print(dec_out.size())
    decode_time = time.perf_counter() - stime
    return encode_time,decode_time
def print_model(model):
    model_size = sum(p.numel() for p in model.parameters()) / 1000000.
    print('Model size: %.2fM' % model_size)

def warmup(warmup_iter, model):
    if warmup_iter > 0:
        print("Running %d warmup iterations" % warmup_iter)
    while warmup_iter != 0:
        # initialize the hid = (h0, c0) here, because traced model needs it 
        h0 = torch.zeros(size=(12,1,1024), dtype = src.dtype)
        hid = (h0, h0)

        enc_out = model.encode(src.unsqueeze(0), hid)[0]
        enc_mask = torch.ones((1, enc_out.size(1)), dtype=torch.uint8)
        for i in range(5):
            l = random.randint(5, 10)
            seq = torch.LongTensor([l] * l)
            # print(seq.size())
            dec_out = model.decode(enc_out, enc_mask, seq.unsqueeze(0))[0]
        warmup_iter -= 1

if __name__ == '__main__':

    run_quant = 0
    run_ipex = 1
    warmup_iter = 10

    if len(sys.argv) < 2:
        print("Please specify an audio file.")
        sys.exit()
    wav_file = sys.argv[1]

    if len(sys.argv) > 2:
        run_quant = int(sys.argv[2])
        run_ipex = int(sys.argv[3])
        if run_quant == run_ipex == 1:
            print("Warning: Dynamic quantization wit IPEX is not available. You may see unreliable results.")

    print("Extract log mel features from: {}".format(wav_file))
    src = extract_fbank(wav_file)
    print("Audio length: {} ms ".format(src.size(0)*10))
    print_precision="FP32"
    model = Seq2Seq(enc_dropout=0.3,enc_dropconnect=0.3,n_vocab=4003, d_input=40, d_enc=1024, n_enc=6, d_dec=1024, n_dec=2,d_emb=512,d_project=300,n_head=1,use_cnn=True,dec_dropout=0.2,dec_dropconnect=0.2,emb_drop=0.15)
    model.eval()
    print('before quantize')
    print_model(model)
    if run_quant==1:
        print('run_quant')
        print("Quantizing %s model using torch.quantization" %print_precision)
        print_precision="INT8"
        model = torch.quantization.quantize_dynamic(
            model,  # the original model
            {
                torch.nn.LSTM,
                torch.nn.Linear,
            },  # a set of layers to dynamically quantize
            dtype=torch.qint8,
        )  # the target dtype for quantized weights

    ## Optimize model with IPEX
    elif run_ipex==1:
        print('run_ipex')
        # Use IPEX for encoder
        # model.encoder = torch.jit.load("quantized_encoder.pt")
        model.encoder.rnn = torch.jit.load("quantized_lstm.pt")
        # Use fbgemm for decoder
        model.decoder = torch.quantization.quantize_dynamic(
            model.decoder,  # the original model
            {
                torch.nn.LSTM,
                torch.nn.Linear,
            },  # a set of layers to dynamically quantize
            dtype=torch.qint8,
        )  # the target dtype for quantized weights


    with torch.no_grad():
        print_model(model)
        print(model)

        warmup(10, model)

        print("Starting encoding and decoding with the %s Seq2Seq model..."%print_precision)
        eval(model,src)
        print("Done.")

    stime = time.perf_counter()
    enc_out = model.encode(src.unsqueeze(0), None)[0]
    print("Encoding: {}".format(time.perf_counter()-stime))
    
    stime = time.perf_counter()
    enc_mask = torch.ones((1, enc_out.size(1)), dtype=torch.uint8)
    for i in range(5):
        l = random.randint(5,10)
        seq = torch.LongTensor([l]*l)
        #print(seq.size())
        dec_out = model.decode(enc_out, enc_mask, seq.unsqueeze(0))[0]
        #print(dec_out.size())
    print("Decoding: {}".format(time.perf_counter()-stime))