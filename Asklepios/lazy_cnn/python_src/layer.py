#from unicodedata import bidirectional
#from unittest.util import _MAX_LENGTH
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, device='cuda:0'):
        #super().__init__()
        super(Encoder, self).__init__()
        self.device = device
        #self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=hidden_dim, padding_idx=0)
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
        nn.init.xavier_normal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)

    def forward(self, x):
        #len_source_sequences = (x.t() > 0).sum(dim=-1)
        #len_source_sequences = len_source_sequences.to('cpu')
        x = self.embedding(x)
        #x = pack_padded_sequence(input=x, lengths=len_source_sequences, enforce_sorted=False)
        h, states = self.gru(x)
        #h, _ = pad_packed_sequence(h)
        return h, states

class Attention(nn.Module):
    def __init__(self, output_dim, hidden_dim, device='cuda:0'):
        #super().__init__()
        super(Attention, self).__init__()
        self.device = device
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.W_a = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.W_c = nn.Parameter(torch.Tensor(hidden_dim * 4, output_dim * 4))
        self.b = nn.Parameter(torch.zeros(output_dim * 4))
        nn.init.xavier_normal_(self.W_a)
        nn.init.xavier_normal_(self.W_c)

    def forward(self, ht, hs, source=None):
        # ht: デコーダの値（Attention前）
        # hs: エンコーダの値

        #スコア関数の計算
        score = torch.einsum('jik,kl->jil', (hs, self.W_a))
        #x = torch.zeros(128*64*128, device=self.device)
        #x = x.reshape(128,64,128)
        #score = torch.einsum('jik,lik->jil', (ht, x))
        score = torch.einsum('jik,lik->jil', (ht, score))

        #加重平均の計算（スコア関数のソフトマックス計算）
        score = score - torch.max(score, dim=-1, keepdim=True)[0]
        score = torch.exp(score)
        #if source is not None:
            #mask_source = source.t().eq(0).unsqueeze(0)#パディング部分を求める
            #score.data.masked_fill_(mask_source, 0)#マスク処理
        a = score / torch.sum(score, dim=-1, keepdim=True)

        #文脈ベクトルの計算
        c = torch.einsum('jik,kil->jil', (a, hs))

        #出力の計算
        h = torch.cat((c, ht), -1)

        #x = torch.zeros(256*256, device=self.device)
        #x = x.reshape(256, 256)

        #t = torch.einsum('jik,kl->jil', (h, x))
        #t = torch.einsum('jik,kl->jil', (h, self.W_c))

        return torch.tanh(torch.einsum('jik,kl->jil', (h, self.W_c)) + self.b)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, device='cuda:0'):
        #super().__init__()
        super(Decoder, self).__init__()
        self.device = device
        #self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=hidden_dim, padding_idx=0)
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=hidden_dim)
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True)
        self.attn = Attention(hidden_dim, hidden_dim)
        self.out = nn.Linear(in_features=hidden_dim * 4, out_features=output_dim)
        nn.init.xavier_normal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x, hs, states, source=None):
        # x     : デコーダへの入力系列
        # hs    : エンコーダの出力値
        # states: エンコーダの状態値
        # source: エンコーダへの入力系列
        x = self.embedding(x)
        ht, states = self.gru(x, states)
        ht = self.attn(ht, hs, source=source)
        y = self.out(ht)
        return y, states

class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, max_length=100, device='cuda:0'):
        #super().__init__()
        super(EncoderDecoder, self).__init__()
        self.device = device
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.encoder = self.encoder.to(device)
        self.decoder = Decoder(hidden_dim, output_dim, num_layers)
        self.decoder = self.decoder.to(device)
        self.max_length = max_length
        self.output_dim = output_dim#32*32*81
        fcl = 256
        self.l1 = nn.Linear(self.max_length * self.output_dim, fcl)
        self.l2 = nn.Linear(fcl, self.output_dim)

    def forward(self, source, target=None, use_teacher_forcing=False):
        batch_size = source.size(1)
        #if target is not None:
            #len_target_sequences = target.size(0)
        #else:
            #len_target_sequences = self.max_length
        len_target_sequences = self.max_length

        hs, states = self.encoder(source)

        #y = torch.ones((1, batch_size), dtype=torch.long, device=self.device)#BOSのIDを設定する（1がBOSのID
        y = torch.full((1, batch_size), fill_value=0, dtype=torch.long, device=self.device)#0はNull Move
        output = torch.zeros((len_target_sequences, batch_size, self.output_dim), device=self.device)
        #output = torch.zeros((batch_size, self.output_dim), device=self.device)

        for t in range(len_target_sequences):

            if use_teacher_forcing == False:
                source = None
            #else:
                #source = source[t]

            out, states = self.decoder(y, hs, states, source=source)
            output[t] = out #Tensorに値を直接代入できる

            if use_teacher_forcing and target is not None:
                #y = target[t].unsqueeze(0) #時系列の次元を追加する
                y = target.unsqueeze(0)
            else:
                y = out.max(-1)[1]

        output = output.reshape(batch_size, self.max_length * self.output_dim)

        output = F.relu(self.l1(output))
        output = self.l2(output)

        return output

class EncoderDecoderValue(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, max_length=100, device='cuda:0'):
        #super().__init__()
        #super().__init__()
        super(EncoderDecoderValue, self).__init__()
        self.device = device
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.encoder = self.encoder.to(device)
        self.decoder = Decoder(hidden_dim, output_dim, num_layers)
        self.decoder = self.decoder.to(device)
        self.max_length = max_length
        self.output_dim = output_dim#32*32*81
        fcl = 256
        self.l1 = nn.Linear(self.max_length * self.output_dim, fcl)
        self.l2 = nn.Linear(fcl, 1)

    def forward(self, source, target=None, use_teacher_forcing=False):
        batch_size = source.size(1)
        #if target is not None:
            #len_target_sequences = target.size(0)
        #else:
            #len_target_sequences = self.max_length
        len_target_sequences = self.max_length

        hs, states = self.encoder(source)

        #y = torch.ones((1, batch_size), dtype=torch.long, device=self.device)#BOSのIDを設定する（1がBOSのID
        y = torch.full((1, batch_size), fill_value=0, dtype=torch.long, device=self.device)#0はNull Move
        output = torch.zeros((len_target_sequences, batch_size, self.output_dim), device=self.device)
        #output = torch.zeros((batch_size, self.output_dim), device=self.device)

        for t in range(len_target_sequences):

            if use_teacher_forcing == False:
                source = None
            #else:
                #source = source[t]

            out, states = self.decoder(y, hs, states, source=source)
            output[t] = out #Tensorに値を直接代入できる

            if use_teacher_forcing and target is not None:
                #y = target[t].unsqueeze(0) #時系列の次元を追加する
                y = target.unsqueeze(0)
            else:
                y = out.max(-1)[1]

        output = output.reshape(batch_size, self.max_length * self.output_dim)

        output = F.relu(self.l1(output))
        output = self.l2(output)
        output = output.reshape(-1)

        return output