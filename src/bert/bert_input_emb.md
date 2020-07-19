* 代码来源于 https://github.com/huggingface/transformers 中的BERT模块
* BERT的输入来源于三个向量相加
    * token emb
        * 字/词向量，可导入预训练的word2vec向量作为初始化权重，也可以重新训练
    * position emb
       * 位置向量，每个token按照所在位置序号对应一个从0到max_seq_length的id，每个id对应一个训练得到的emb
       * 也可以按照transformer的位置编码，按照正弦函数的位置依赖关系，人工设计固定的位置向量
    * segment emb
       * 段落向量，由于bert是采用next sentence prediction的方式进行训练，因此需要利用不同句子间的顺序关系，该向量和位置向量一样，也是先随机初始化后通过训练得到
       * 我们给第一句话的段落向量全部置为0，第二句话置为1，如果在非训练阶段使用BERT时，只需要将该段落向量全部置为0即可

```python
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```