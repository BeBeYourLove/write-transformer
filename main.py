from Embeddings import Embeddings
import torch
from torch.autograd import Variable

from EncoderLayer import EcoderLayer
from LayerNorm import LayerNorm
from MultiHeadedAttention import MultiHeadedAttention
from PositionEmbedding import PositionEmbedding
from PositionwiseFeedForward import PositionwiseEmbedding
from SubLayerConnection import SubLayerConnection
from TongHeadedAttention import TongHeadedAttention

if __name__ == '__main__':
    head = 8
    embedding_dim = 512
    d_model = 512
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
    dropout = 0.2
    self_attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseEmbedding(d_model, d_ff, dropout)
    mask = Variable(torch.zeros(8, 4, 4))

    el = EcoderLayer(d_model, self_attn, ff, dropout)
    el_result = el(x, mask)
    print(el_result.shape)
    print(el_result)