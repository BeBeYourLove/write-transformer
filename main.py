from Embeddings import Embeddings
import torch
from torch.autograd import Variable

from LayerNorm import LayerNorm
from MultiHeadedAttention import MultiHeadedAttention
from PositionEmbedding import PositionEmbedding
from PositionwiseFeedForward import PositionwiseEmbedding
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
    query = key = value = res
    # print("query等输入的维度为:", query.shape)
    mask = Variable(torch.zeros(8, 4, 4))
    MultiAttention = MultiHeadedAttention(head, embedding_dim, dropout)
    mha_res = MultiAttention(query, key, value, mask)
    ff = PositionwiseEmbedding(d_model, d_ff, dropout)
    res = ff(mha_res)
    features = d_model = 512
    eps = 1e-6
    ln = LayerNorm(features, eps)
    res = ln(res)
    print(res.shape)
    print(res)

