import torch.nn as nn
import torch as t
import torch.nn.functional as F
import math
import hyperparams as hp
from text.symbols import symbols
import numpy as np
import copy
from collections import OrderedDict

# 包含所有的模型方法
def clones(module, N):
    """对传入的module深度复制n份，并放在一个modulelist中"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Linear(nn.Module):
    # Linear Module 定义线性全连接层，使用xavier_uniform_进行初始化
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class Conv(nn.Module):
    """
    Convolution Module 定义一维CNN，使用xavier_uniform_进行初始化
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


class EncoderPrenet(nn.Module):
    # 编码器处的预处理网络
    def __init__(self, embedding_size, num_hidden):
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(len(symbols), embedding_size, padding_idx=0)
        self.conv1 = Conv(in_channels=embedding_size,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        self.conv3 = Conv(in_channels=num_hidden,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')
        self.batch_norm1 = nn.BatchNorm1d(num_hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_hidden)
        self.batch_norm3 = nn.BatchNorm1d(num_hidden)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.projection = Linear(num_hidden, num_hidden)

    def forward(self, input_):
        input_ = self.embed(input_)  # b t d
        input_ = input_.transpose(1, 2)  # b d t
        input_ = self.dropout1(t.relu(self.batch_norm1(self.conv1(input_)))) 
        input_ = self.dropout2(t.relu(self.batch_norm2(self.conv2(input_)))) 
        input_ = self.dropout3(t.relu(self.batch_norm3(self.conv3(input_)))) 
        input_ = input_.transpose(1, 2) # b t d
        input_ = self.projection(input_) 

        return input_


class FFN(nn.Module):
    def __init__(self, num_hidden):
        super(FFN, self).__init__()
        self.w_1 = Conv(num_hidden, num_hidden * 4, kernel_size=1, w_init='relu')
        self.w_2 = Conv(num_hidden * 4, num_hidden, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, input_):
        x = input_.transpose(1, 2) 
        x = self.w_2(t.relu(self.w_1(x))) 
        x = x.transpose(1, 2) 

        x = x + input_ 

        # dropout
        # x = self.dropout(x) 

        x = self.layer_norm(x) 

        return x


class PostConvNet(nn.Module):
    # (mel --> mel) 解码器处的后处理网络
    def __init__(self, num_hidden):
        super(PostConvNet, self).__init__()
        self.conv1 = Conv(in_channels=hp.num_mels * hp.outputs_per_step,
                          out_channels=num_hidden,
                          kernel_size=5,
                          padding=4,
                          w_init='tanh')
        self.conv_list = clones(Conv(in_channels=num_hidden,
                                     out_channels=num_hidden,
                                     kernel_size=5,
                                     padding=4,
                                     w_init='tanh'), 3)
        self.conv2 = Conv(in_channels=num_hidden,
                          out_channels=hp.num_mels * hp.outputs_per_step,
                          kernel_size=5,
                          padding=4)
        self.batch_norm_list = clones(nn.BatchNorm1d(num_hidden), 3)
        self.pre_batchnorm = nn.BatchNorm1d(num_hidden)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=0.1) for _ in range(3)])

    def forward(self, input_, mask=None):
        # Causal Convolution (for auto-regressive)
        # 因为在构建conv时，kernel_size为5，但padding为4，输出的维度是比num_hidden多4，故都不取最后的四条数据
        input_ = self.dropout1(t.tanh(self.pre_batchnorm(self.conv1(input_)[:, :, :-4])))
        for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
            input_ = dropout(t.tanh(batch_norm(conv(input_)[:, :, :-4])))
        input_ = self.conv2(input_)[:, :, :-4]
        return input_


class MultiheadAttention(nn.Module):
    # Multihead attention mechanism (dot attention) 多头自注意力层
    def __init__(self, num_hidden_k):
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query, mask=None, query_mask=None):
        # 获得注意力分数
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)
        # Masking to ignore padding
        if mask is not None:
            attn = attn.masked_fill(mask, -2 ** 32 + 1) # 将 attn 中与 mask 为 1 对用的位置的值用 -2 ** 32 + 1 填充
            attn = t.softmax(attn, dim=-1)
        else:
            attn = t.softmax(attn, dim=-1)

        if query_mask is not None:
            attn = attn * query_mask # 将 attn 与 query_mask 中的值按位相乘

        # Dropout
        # attn = self.attn_dropout(attn)
        
        result = t.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    # Attention Network 注意力层
    def __init__(self, num_hidden, h=4):
        # num_hidden: 隐藏单元维度
        # h: 自注意力头的个数 
        super(Attention, self).__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h # 每个自注意力头的维度数
        self.h = h 
        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)
        self.multihead = MultiheadAttention(self.num_hidden_per_attn)
        self.residual_dropout = nn.Dropout(p=0.1)
        self.final_linear = Linear(num_hidden * 2, num_hidden)
        self.layer_norm_1 = nn.LayerNorm(num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):
        '''
        :param memory:相当于key和value，[bsz,memory_len,num_hidden]
        :param decoder_input:相当于query，[bsz,decoder_input_len,num_hidden]
        :param mask：[bsz,decoder_input_len,memory_len]
        :param query_mask，对问句序列的掩码，即query的掩码，[bsz,decoder_input_len]
        '''
        batch_size = memory.size(0)
        seq_k = memory.size(1)
        seq_q = decoder_input.size(1)
        
        # Repeat masks h times
        if query_mask is not None:
            # [bsz,seq_q]->[bsz,seq_q,1]->[bsz,seq_q,seq_k],padding部分为0，非padding部分为1
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)
            query_mask = query_mask.repeat(self.h, 1, 1)
        if mask is not None:
            # [bsz,seq_p,seq_k]->[h*bsz,seq_p,seq_k]
            mask = mask.repeat(self.h, 1, 1)

        # Make multihead 初始化key、value和query,[bsz,seq_k,h,num_hidden_per_attn]
        key = self.key(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(decoder_input).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        # 先把头这个维度放在最前面，然后再把头的维度和batch_size维度两个维度拉直
        # [bsz,seq_k,h,num_hidden_per_attn]->[h,bsz,seq_k,num_hidden_per_attn]->[h*bsz,seq_k,num_hidden_per_attn]
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        # [bsz,seq_k,h,num_hidden_per_attn]->[h,bsz,seq_k,num_hidden_per_attn]->[h*bsz,seq_k,num_hidden_per_attn]
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        # [bsz,seq_q,h,num_hidden_per_attn]->[h,bsz,seq_q,num_hidden_per_attn]->[h*bsz,seq_q,num_hidden_per_attn]
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector result的维度是[h*bsz,seq_q,num_hidden_per_attn]，大小没有变化；attns的维度是[h*bsz,seq_q,seq_k]
        result, attns = self.multihead(key, value, query, mask=mask, query_mask=query_mask)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn) # [h,bsz,seq_q,num_hidden_per_attn]
        # result的维度是[bsz,seq_q,num_hidden]，与decoder_input的大小一致，没有变化
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)
        
        result = t.cat([decoder_input, result], dim=-1) # [bsz,seq_k,2*num_hidden]
        
        result = self.final_linear(result) # [bsz,seq_k,num_hidden]

        result = result + decoder_input

        # result = self.residual_dropout(result)

        result = self.layer_norm_1(result)

        return result, attns
    

class Prenet(nn.Module):
    # 解码器处预处理网络
    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        # param input_size: 输入维度
        # param hidden_size: 隐藏单元的维度
        # param output_size: 输出维度
        super(Prenet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', Linear(self.input_size, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(p)),
             ('fc2', Linear(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(p)),
        ]))

    def forward(self, input_):
        out = self.layer(input_)
        return out
    
class CBHG(nn.Module):
    """
    CBHG Module 标准的CBHG模块，将mel->linear
    """
    def __init__(self, hidden_size, K=16, projection_size = 256, num_gru_layers=2, max_pool_kernel_size=2, is_post=False):
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.convbank_list = nn.ModuleList() # 存放K个一维卷积
        self.convbank_list.append(nn.Conv1d(in_channels=projection_size,
                                                out_channels=hidden_size,
                                                kernel_size=1,
                                                padding=int(np.floor(1/2))))

        for i in range(2, K+1):
            self.convbank_list.append(nn.Conv1d(in_channels=hidden_size,
                                                out_channels=hidden_size,
                                                kernel_size=i,
                                                padding=int(np.floor(i/2))))

        self.batchnorm_list = nn.ModuleList() # 存放K个batchnorm
        for i in range(1, K+1):
            self.batchnorm_list.append(nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K
        
        self.conv_projection_1 = nn.Conv1d(in_channels=convbank_outdim,
                                             out_channels=hidden_size,
                                             kernel_size=3,
                                             padding=int(np.floor(3 / 2)))
        self.conv_projection_2 = nn.Conv1d(in_channels=hidden_size,
                                               out_channels=projection_size,
                                               kernel_size=3,
                                               padding=int(np.floor(3 / 2)))
        self.batchnorm_proj_1 = nn.BatchNorm1d(hidden_size)

        self.batchnorm_proj_2 = nn.BatchNorm1d(projection_size)


        self.max_pool = nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = nn.GRU(self.projection_size, self.hidden_size // 2, num_layers=num_gru_layers,
                          batch_first=True,
                          bidirectional=True) # 双向GRU


    def _conv_fit_dim(self, x, kernel_size=3):
        if kernel_size % 2 == 0:
            return x[:,:,:-1]
        else:
            return x

    def forward(self, input_):

        input_ = input_.contiguous()
        batch_size = input_.size(0)
        total_length = input_.size(-1)

        convbank_list = list()
        convbank_input = input_

        for k, (conv, batchnorm) in enumerate(zip(self.convbank_list, self.batchnorm_list)):
            convbank_input = t.relu(batchnorm(self._conv_fit_dim(conv(convbank_input), k+1).contiguous()))
            convbank_list.append(convbank_input)

        conv_cat = t.cat(convbank_list, dim=1)

        conv_cat = self.max_pool(conv_cat)[:,:,:-1]

        conv_projection = t.relu(self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat))))
        conv_projection = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_projection))) + input_

        highway = self.highway.forward(conv_projection.transpose(1,2))
        
        self.gru.flatten_parameters()
        out, _ = self.gru(highway)

        return out


class Highwaynet(nn.Module):
    def __init__(self, num_units, num_layers=4):
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(Linear(num_units, num_units))
            self.gates.append(Linear(num_units, num_units))

    def forward(self, input_):
        out = input_

        for fc1, fc2 in zip(self.linears, self.gates):

            h = t.relu(fc1.forward(out))
            t_ = t.sigmoid(fc2.forward(out))

            c = 1. - t_
            out = h * t_ + out * c

        return out
