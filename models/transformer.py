# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        # encoder和decoder的参数基本是相同的

        # encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)

        decoder_norm = nn.LayerNorm(d_model)

        # return_intermediate 就是把前5层的值也返回
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model

        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        scr: (bs 256 H W)
        mask: (bs H W)
        query_embed: (100,256)
        pos_embed: (bs 256 H W)
        """

        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        # HW 推平 -> HW,bs,256
        src = src.flatten(2).permute(2, 0, 1)
        # 变成与上面的维度相同
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        # 传进来的是query_embed.weight
        # query_embed 是给decoder使用的
        # 100， 256 中间加个维度，并将这个维度重复 bs 次 -> 100,bs,256
        # [100,256] --unsqueeze(1)--> [100,1,256] --repeat--> [100,bs,256]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # hw 推平
        mask = mask.flatten(1)

        # [100,bs,256]
        tgt = torch.zeros_like(query_embed)

        # feature和位置编码 进入encoder网络 输出 memory: HW,bs,256
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # tgt, encoder的输出，位置编码，以及query输入到decoder
        # hs [6(decoder layer number),100,bs,256]
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        # memory 变换成 bs,c,h,w的形式
        # memory 是encoder的输出
        # hs 是decoder的输出
        # hs [6,100,bs,256] --transpose--> [6,bs,100,256]
        # memory [hw,bs,256] --permute--> [bs,256,hw] --view--> [bs,256,h,w]
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


# ----------------------------------------------------------------------------------------------------------------------

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # 复制几层
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        # 6层
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        # 最后经过norm
        if self.norm is not None:
            output = self.norm(output)

        return output


# encoder
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        d_model: 512 是encoder中的宽度
        2048 FNN的宽度
        """
        super().__init__()
        # 修改一下顺序，在print的时候会好看些

        # torch内置，直接使用
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout2 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """
        norm在attention和mlp之后
        src: (hw,3,256)
        """
        # q k 融合位置编码，value不使用位置编码
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 残差的就直接加进去了
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """
        norm在attention和mlp之前
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


# ----------------------------------------------------------------------------------------------------------------------

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # 复制几层
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                # tgt = torch.zeros_like(query_embed)
                # 初始时是全是0，与query_embed大小相同
                tgt,
                # memory 是 encoder的输出
                memory,
                # 分割任务使用的
                tgt_mask: Optional[Tensor] = None,
                # 分割任务使用的
                memory_mask: Optional[Tensor] = None,
                # 分割任务使用的
                tgt_key_padding_mask: Optional[Tensor] = None,
                # 检测任务会提供这个
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        """
        tgt: [100,bs,256]
        memory: [hw,bs,256]
        pos: [hw,bs,256]
        query_pos: [100,bs,256[
        """

        # 对于第一层的decoder, 它的输入object query是[100,bs,256]的zero tensor
        # 但是对于第二层及以后的decoder，他的输入上是上一次的decoder的输出结果
        # 但是不管第几层的decoder，它们使用的memory(encoder结构的输出) 是相同的，都是最后一层encoder的输出
        output = tgt
        # 保留中间层的输出
        intermediate = []

        for layer in self.layers:
            # [100,bs,256]
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            # 每一个中间层计算的结果需要返回
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        # 最后经过norm
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                # 弹出最后一个，加上上面经过norm的
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        # decoder encoder的初始化参数相同
        super().__init__()

        # 调整了顺序

        # 这是论文中decoder中下面那个
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 这是论文中decoder中上面那个
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 就是两个tensor相加
        return tensor if pos is None else tensor + pos

    # norm在操作后
    def forward_post(self,
                     # 第一层的tgt是 tgt = torch.zeros_like(query_embed)
                     tgt,
                     # encoder的输出
                     memory,
                     # 检测任务没有
                     tgt_mask: Optional[Tensor] = None,
                     # 检测任务没有
                     memory_mask: Optional[Tensor] = None,
                     # 检测任务没有
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     # 检测任务只有这个
                     memory_key_padding_mask: Optional[Tensor] = None,
                     # 位置嵌入
                     pos: Optional[Tensor] = None,
                     # 100的那个query, 就是query_embed
                     query_pos: Optional[Tensor] = None):
        """
        tgt: [100,bs,256]
        memory: [hw,bs,256]
        pos: [hw,bs,256]
        query_pos: [100,bs,256]
        """

        # 这两个attention跟论文上的图中画的路线是一样的
        # query_pos 就是 query_embed
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q,
                              k,
                              value=tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # 第一个dropout
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 各项值的组合方式与论文中的图像是一致的
        tgt2 = self.multihead_attn(
            # 第二个attention的query是第一个attention的输出
            query=self.with_pos_embed(tgt, query_pos),
            # 两个参数，跟上面的都不同
            # 使用了encoder的输出memory，以及位置编码
            key=self.with_pos_embed(memory, pos),
            # 上面是tgt，这个是memory
            # encoder的输出memory是decoder中第二个attention的value
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]

        # 第二个dropout
        # 这里的tgt是第一个attention的输出 加上 第二个attention的输出
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # 第三个dropout
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    # norm在操作前
    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            # norm在sa之前
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        # norm在sa之后
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


# ----------------------------------------------------------------------------------------------------------------------

def _get_clones(module, N):
    # encoder，decoder内部的那几层结构都是相同的
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        # 返回decoder的中间层结果
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    # 三种激活函数
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
