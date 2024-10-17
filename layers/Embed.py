import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class TruncateModule(nn.Module):
    def __init__(self, target_length):
        super(TruncateModule, self).__init__()
        self.target_length = target_length

    def forward(self, x, truncate_length):
        return x[: ,: ,:truncate_length]


def PositionalEncoding(q_len, d_model, normalize=False):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        print(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type='GRU', ksize=3):
        super(LocalRNN, self).__init__()
        """
        LocalRNN structure
        """
        self.ksize = ksize
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(output_dim, output_dim, batch_first=True)

        # To speed up
        idx = [i for j in range(self.ksize-1,10000,1) for i in range(j-(self.ksize-1),j+1,1)]
        self.select_index = torch.LongTensor(idx).cuda()
        self.zeros = torch.zeros((self.ksize-1, input_dim)).cuda()

    def forward(self, x):
        nbatches, l, input_dim = x.shape
        # x: [bs x patch_num x d_model] → [b x seq_len x ksize x d_model]
        x = self.get_K(x)
        batch, l, ksize, d_model = x.shape
        h = self.rnn(x.view(-1, self.ksize, d_model))[0][:,-1,:]
        return h.view(batch, l, d_model)

    def get_K(self, x):
        batch_size, l, d_model = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1).to(x.device)
        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.select_index[:self.ksize*l].to(x.device))
        key = key.reshape(batch_size, l, self.ksize, -1)
        return key

class PositionalEmbedding(nn.Module):
    def __init__(self, q_len=5000, d_model=128, pos_embed_type='sincos', learnable=False, r_layers=1, c_in=21, scale=1):
        super(PositionalEmbedding, self).__init__()
        self.pos_embed_type = pos_embed_type
        self.learnable = learnable
        self.scale = scale
        if pos_embed_type == None:
            W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pos_embed_type == 'zero':
            W_pos = torch.empty((q_len, 1))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pos_embed_type == 'zeros':
            W_pos = torch.empty((q_len, d_model))
            nn.init.uniform_(W_pos, -0.02, 0.02)
        elif pos_embed_type == 'normal' or pos_embed_type == 'gauss':
            W_pos = torch.zeros((q_len, 1))
            torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
        elif pos_embed_type == 'uniform':
            W_pos = torch.zeros((q_len, 1))
            nn.init.uniform_(W_pos, a=0.0, b=0.1)
        elif pos_embed_type == 'random': W_pos = torch.rand(c_in, q_len, d_model)
        elif pos_embed_type == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
        elif pos_embed_type == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
        elif pos_embed_type == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
        elif pos_embed_type == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
        elif pos_embed_type == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
        elif pos_embed_type == 'localrnn': W_pos = nn.Sequential(*[LocalRNN(d_model, d_model) for _ in r_layers])
        elif pos_embed_type == 'rnn': W_pos = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=r_layers)
        else: raise ValueError(f"{pos_embed_type} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
        if 'rnn' in pos_embed_type:
            self.pos = W_pos
        else:
            W_pos = W_pos.unsqueeze(0)  # [1, L, D] or [1, C, L, D]
            if learnable:
                self.pos = nn.Parameter(W_pos, requires_grad=learnable)
            else:
                self.register_buffer('pos', W_pos)
                self.pos = W_pos

    def forward(self, x):
        if 'rnn' in self.pos_embed_type:
            output, _ = self.pos(x)
            return output
        # pos generated for individual variable
        if self.pos.dim()>3:
            batch_size = x.size(0) // self.pos.size(1)
            self.pos = self.pos.repeat(batch_size, 1, 1, 1)
            self.pos = torch.reshape(self.pos, (-1, self.pos.shape[2], self.pos.shape[3]))
            return self.pos
        else:
            return self.pos[:, self.scale-1:x.size(1)*self.scale:self.scale]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x_mark = x_mark.permute(0, 2, 1)
            x = self.value_embedding(torch.cat([x, x_mark], 1))
        return self.dropout(x)



class PatchEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, patch_len, stride, dropout,
                 process_layer=None, 
                 pos_embed_type=None, learnable=False, r_layers=1,
                 ch_ind=0, 
                 deform_patch=False):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.ch_ind = ch_ind
        self.process_layer = process_layer
        self.pos_embed_type = pos_embed_type

        # Deformable Patch
        self.deform_patch = deform_patch
        if self.deform_patch:
            self.patch_num = int((seq_len - patch_len)/stride + 1)
            self.patch_shift_linear = nn.Linear(seq_len, self.patch_num * 3)
            self.box_coder = pointwhCoder(input_size=seq_len, 
                                          patch_count=self.patch_num, 
                                          weights=(1.,1.,1.), 
                                          pts=self.patch_len, 
                                          tanh=True, 
                                          wh_bias=torch.tensor(5./3.).sqrt().log(),
                                          deform_range=0.5)

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model)

        # Positional embedding
        if self.pos_embed_type is not None:
            self.position_embedding = PositionalEmbedding(seq_len, d_model, pos_embed_type, learnable, r_layers, patch_len)
            
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        # x: [B, M, L], x_mark: [B, 4, L]
        batch_size, n_vars, _ = x.shape

        if self.deform_patch:
            x_lfp = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)                       # [B, M, N, P]
            x_lfp = x_lfp.permute(0,1,3,2)                                                              # [B, M, P, N]
            anchor_shift = self.patch_shift_linear(x).view(batch_size * n_vars, self.patch_num, 3)
            sampling_location_1d = self.box_coder(anchor_shift) # B*C, self.patch_num,self.patch_len, 1
            add1d = torch.ones(size=(batch_size * n_vars, self.patch_num, self.patch_len, 1)).float().to(sampling_location_1d.device)
            sampling_location_2d = torch.cat([sampling_location_1d, add1d],dim=-1)
            x = x.reshape(batch_size * n_vars, 1, 1, self.seq_len)
            patch = F.grid_sample(x, sampling_location_2d, mode='bilinear', padding_mode='border', align_corners=False).squeeze(1)  # B*C, self.patch_num,self.patch_len
            x = patch.reshape(batch_size, n_vars, self.patch_num, self.patch_len) # [bs x nvars x patch_num x patch_len]
            PaEN_Loss = cal_PaEn(x_lfp, x.permute(0,1,3,2), 0.01, 0.1)
        else:
            x = self.process_layer(x)
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        if x_mark is not None and not self.ch_ind:
            n_vars+=4
            x_mark = x_mark.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            x = torch.cat([x, x_mark], dim=1) # [B, (M+4), N, P]
        
        # Input value embedding
        x = self.value_embedding(x) # [B, M, N, D]
        
        if self.ch_ind == 0:
            x = x.permute(0, 2, 1, 3) # [B, N, M, D]
        x = torch.reshape(x, (-1, x.shape[2], x.shape[3]))

        if self.pos_embed_type is not None:
            x = x + self.position_embedding(x)
        
        if self.deform_patch:
            return self.dropout(x), (n_vars, PaEN_Loss)
        else:
            return self.dropout(x), n_vars

def generate_pairs(n):
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append([i, j])
    return np.array(pairs)


def cal_PSI(x, r):
    #[bs x nvars x patch_len x patch_num]
    x = x.permute(0,1,3,2)
    batch, n_vars, patch_num, patch_len = x.shape
    x = x.reshape(batch*n_vars, patch_num, patch_len)
    # Generate all possible pairs of patch_num indices within each batch
    pairs = generate_pairs(patch_num)
    # Calculate absolute differences between pairs of sequences
    abs_diffs = torch.abs(x[:, pairs[:, 0], :] - x[:, pairs[:, 1], :])
    # Find the maximum absolute difference for each pair of sequences
    max_abs_diffs = torch.max(abs_diffs, dim=-1).values 
    max_abs_diffs = max_abs_diffs.reshape(-1,patch_num,patch_num-1)
    # Count the number of pairs with max absolute difference less than r
    c = torch.log(1+torch.mean((max_abs_diffs < r).float(),dim=-1))
    psi = torch.mean(c,dim=-1)
    return psi

def cal_PaEn(lfp,lep,r,lambda_):
    psi_lfp = cal_PSI(lfp,r)
    psi_lep = cal_PSI(lep,r)
    psi_diff = psi_lfp - psi_lep
    lep = lep.permute(0,1,3,2)
    batch, n_vars, patch_num, patch_len = lep.shape    
    lep = lep.reshape(batch*n_vars, patch_num, patch_len)
    sum_x = torch.sum(lep, dim=[-2,-1])
    PaEN_loss = torch.mean(sum_x*psi_diff)*lambda_ # update parameters with REINFORCE
    return PaEN_loss

class pointCoder(nn.Module):
    def __init__(self, input_size, patch_count, weights=(1., 1.,1.), tanh=True):
        super().__init__()
        self.input_size = input_size
        self.patch_count = patch_count
        self.weights = weights
        #self._generate_anchor()
        self.tanh = tanh

    def _generate_anchor(self, device="cpu"):
        anchors = []
        patch_stride_x = 2. / self.patch_count
        for i in range(self.patch_count):
                x = -1+(0.5+i)*patch_stride_x
                anchors.append([x])
        anchors = torch.as_tensor(anchors)
        self.anchor = torch.as_tensor(anchors, device=device)
        #self.register_buffer("anchor", anchors)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, pts, model_offset=None):
        assert model_offset is None
        self.boxes = self.decode(pts)
        return self.boxes

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel = 1./self.patch_count
        wx, wy = self.weights

        dx = F.tanh(rel_codes[:, :, 0]/wx) * pixel if self.tanh else rel_codes[:, :, 0]*pixel / wx
        dy = F.tanh(rel_codes[:, :, 1]/wy) * pixel if self.tanh else rel_codes[:, :, 1]*pixel / wy

        pred_boxes = torch.zeros_like(rel_codes)

        ref_x = boxes[:,0].unsqueeze(0)
        ref_y = boxes[:,1].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x
        pred_boxes[:, :, 1] = dy + ref_y
        pred_boxes = pred_boxes.clamp_(min=-1., max=1.)

        return pred_boxes

    def get_offsets(self):
        return (self.boxes - self.anchor) * self.input_size

class pointwhCoder(pointCoder):
    def __init__(self, input_size, patch_count, weights=(1., 1.,1.), pts=1, tanh=True, wh_bias=None, deform_range=0.5):
        super().__init__(input_size=input_size, patch_count=patch_count, weights=weights, tanh=tanh)
        self.patch_pixel = pts
        self.wh_bias = None
        if wh_bias is not None:
            self.wh_bias = nn.Parameter(torch.zeros(2) + wh_bias)
        self.deform_range = deform_range
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, boxes):
        self._generate_anchor(device=boxes.device)
        # print(boxes.shape)
        # print(self.wh_bias.shape)
        if self.wh_bias is not None:
            boxes[:, :, 1:] = boxes[:, :, 1:] + self.wh_bias
        self.boxes = self.decode(boxes)
        points = self.meshgrid(self.boxes)
        return points

    def decode(self, rel_codes):
        # print ('xyxy decoding')
        boxes = self.anchor
        pixel_x = 2./self.patch_count # patch_count=in_size//stride 这里应该用2除而不是1除 得到pixel_x是两个patch中点的原本距离
        wx,  ww1,ww2 = self.weights

        dx = F.tanh(rel_codes[:, :, 0]/wx) * pixel_x/4 if self.tanh else rel_codes[:, :, 0]*pixel_x / wx #中心点不会偏移超过patch_len

        dw1 = F.relu(F.tanh(rel_codes[:, :, 1]/ww1)) * pixel_x*self.deform_range + pixel_x # 中心点左边长度在[stride,stride+1/4*stride]，右边同理
        dw2 = F.relu(F.tanh(rel_codes[:, :, 2]/ww2)) * pixel_x*self.deform_range + pixel_x #
        # dw = 

        pred_boxes = torch.zeros((rel_codes.shape[0],rel_codes.shape[1],rel_codes.shape[2]-1)).to(rel_codes.device)

        ref_x = boxes[:,0].unsqueeze(0)

        pred_boxes[:, :, 0] = dx + ref_x - dw1
        pred_boxes[:, :, 1] = dx + ref_x + dw2
        pred_boxes = pred_boxes.clamp_(min=-1., max=1.)

        return pred_boxes

    def meshgrid(self, boxes):
        B = boxes.shape[0]
        xs= boxes
        xs = torch.nn.functional.interpolate(xs, size=self.patch_pixel, mode='linear', align_corners=True)
        results = xs
        results = results.reshape(B, self.patch_count,self.patch_pixel, 1)
        #print((1+results[0])/2*336)
        return results
