import torch
import torch.nn as nn

# 使用EncoderDecoder类来实现transformer的结构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """
        初始化参数，encoder是编码器，decoder是解码器，source_embed源数据嵌入函数，target_embed目标数据的嵌入函数，
        generator最后生成器的对象。
        """
        super(EncoderDecoder, self).__init__()
        # 将参数传入到类中
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """
        source代表源数据，target代表目标数据，source_mask和target_mask是对应的掩码张量。
        """
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        """
        将编码器的操作独立出来，方便操作。
        """
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        """
        需注意这里的参数需对应Decoder中的参数顺序
        """
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)
