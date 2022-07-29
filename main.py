import copy

from DecoderLayer import DecoderLayer
from Embeddings import Embeddings
import torch
import torch.nn as nn
from torch.autograd import Variable

from EncoderDecoder import EncoderDecoder
from Generator import Generator
from EncoderLayer import EcoderLayer
from LayerNorm import LayerNorm
from MultiHeadedAttention import MultiHeadedAttention
from PositionEmbedding import PositionEmbedding
from PositionwiseFeedForward import PositionwiseFeedForward
from SubLayerConnection import SubLayerConnection
from TongHeadedAttention import TongHeadedAttention
from Ecoder import Encoder
from Decoder import Decoder

if __name__ == '__main__':
    head = 8
    embedding_dim = 512
    d_model = 512
    size = 512
    d_ff = 64
    dropout = 0.2
    max_len = 60
    vocab = 1000
    embed = Embeddings(d_model, vocab)
    ten = torch.tensor([[234, 123, 543, 122], [235, 124, 789, 567]])
    x = embed(ten)
    pe = PositionEmbedding(d_model, dropout, max_len)
    res = pe(x)
    # No SubLayerNorm
    # query = key = value = res
    # # print("query等输入的维度为:", query.shape)
    # mask = Variable(torch.zeros(8, 4, 4))
    # MultiAttention = MultiHeadedAttention(head, embedding_dim, dropout)
    # mha_res = MultiAttention(query, key, value, mask)
    # ff = PositionwiseEmbedding(d_model, d_ff, dropout)
    # res = ff(mha_res)
    # features = d_model = 512
    # eps = 1e-6
    # ln = LayerNorm(features, eps)
    # res = ln(res)
    # print(res.shape)
    # print(res)

    # Have a SubLayerNorm
    # mask = Variable(torch.zeros(8, 4, 4))
    # # 设置子层中是多头注意力
    # self_attn = MultiHeadedAttention(head, embedding_dim, dropout)
    # sublayer = lambda x: self_attn(x, x, x, mask)
    # sc = SubLayerConnection(d_model, dropout)
    # sc_res = sc(res, sublayer)
    # print(sc_res.shape)
    # print(sc_res)

    # test EcoderLayer
    # dropout = 0.2
    # self_attn = MultiHeadedAttention(head, d_model)
    # ff = PositionwiseEmbedding(d_model, d_ff, dropout)
    # mask = Variable(torch.zeros(8, 4, 4))
    #
    # el = EcoderLayer(d_model, self_attn, ff, dropout)
    # el_result = el(x, mask)
    # print(el_result.shape)
    # print(el_result)

    """test Ecoder编码器"""
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    layer = EcoderLayer(d_model, c(attn), c(ff), dropout)
    # 设置编码器层的个数N
    N = 8
    mask = Variable(torch.zeros(8, 4, 4))
    en = Encoder(layer, N)
    en_res = en(res, mask)
    # print(en_res.shape)
    # print(en_res)

    """test DecoderLayer解码层"""
    """这里测试DecoderLayer需要用到编码器"""
    self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    memory = en_res
    source_mask = target_mask = mask
    dl = DecoderLayer(size, c(self_attn), c(src_attn), c(ff), dropout)
    # dl_res = dl(x, memory, source_mask, target_mask)
    # print(dl_res.shape)
    # print(dl_res)

    """test Decoder解码器"""
    """这里需要用到解码层和编码器"""
    de = Decoder(dl, N)
    de_res = de(x, memory, source_mask, target_mask)
    # print(de_res.shape)
    # print(de_res)

    """test Generator输出层"""
    """这里需要用到解码器和编码器"""
    gen = Generator(d_model, vocab)
    gen_res = gen(de_res)
    # print(gen_res.shape)
    # print(gen_res)

    """最后整个DecoderEncoder模型总体测试"""
    encoder = en
    decoder = de
    source_embed = nn.Embedding(vocab, d_model)
    target_embed = nn.Embedding(vocab, d_model)
    generator = gen
    # 假设源数据与目标数据相同，实际当中其实并不一样
    source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 2]]))
    # 同样实际上source_mask和target_mask也不会一样
    source_mask = target_mask = Variable(torch.zeros(8, 4, 4))

    ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
    ed_result = ed(source, target, source_mask, target_mask)
    print(ed_result.shape)
    print(ed_result)
