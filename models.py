import torch
import tokenizers

def collate_for_mlp(list_of_samples):
    """ Collate function that creates batches of flat docs tensor and offsets """
    offset = 0
    flat_docs, offsets, labels = [], [], []
    for doc, label in list_of_samples:
        if isinstance(doc, tokenizers.Encoding):
            doc = doc.ids
        offsets.append(offset)
        flat_docs.extend(doc)
        labels.append(label)
        offset += len(doc)
    return torch.tensor(flat_docs), torch.tensor(offsets), torch.tensor(labels)


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self,vocab_size, num_classes, num_hidden_layers=1, hidden_size=1024, hidden_act='relu',
                 dropout=0.5, idf=None, bow_aggregation=None, mode='mean', pretrained_embedding=None, freeze=True, embedding_dropout=0.5):

        torch.nn.Module.__init__(self)

        self.activation = getattr(torch.nn.functional, hidden_act)
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        self.dropout = torch.nn.Dropout(dropout)
        self.layers = torch.nn.ModuleList()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.bow_aggregation = bow_aggregation
        if bow_aggregation=='tfidf':
            mode='sum'
        self.idf = idf

        if pretrained_embedding is not None:
            self.embedding = torch.nn.EmbeddingBag.from_pretrained(pretrained_embedding, freeze=freeze, mode=mode)
            embedding_size = pretrained_embedding.size(1)
            self.embedding_is_pretrained = True
        else:
            assert vocab_size is not None
            self.embedding = torch.nn.EmbeddingBag(vocab_size, hidden_size, mode=mode)
            embedding_size = hidden_size
            self.embedding_is_pretrained = False

        if num_hidden_layers > 0:
            self.layers.append(torch.nn.Linear(embedding_size, hidden_size))
            for i in range(num_hidden_layers):
                self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(torch.nn.Linear(hidden_size, num_classes))
        else:
            self.layers.append(torch.nn.Linear(embedding_size, num_classes))

    def forward(self, input, offset, labels = None):
        if self.bow_aggregation=='tfidf':
            idf_weights = self.idf[input]
        else:
            idf_weights = None

        h = self.embedding(input, offset, per_sample_weights = idf_weights)

        if self.idf is not None:
            h = h / torch.linalg.norm(h, dim=1, keepdim=True)

        if not self.embedding_is_pretrained:
            h = self.activation(h)
        h = self.embedding_dropout(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)

        if labels is not None:
            loss = self.loss_function(h, labels)
            return loss, h

        return h
