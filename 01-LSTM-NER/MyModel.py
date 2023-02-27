import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, token_num, class_num, embedding_num, hidden_num, bi=True):
        super().__init__()

        self.pred = None
        # jy: Embedding(1794, 101)
        self.embedding = nn.Embedding(token_num, embedding_num)
        # jy: LSTM(101, 107, batch_first=True)
        self.lstm = nn.LSTM(embedding_num, hidden_num, batch_first=True, bidirectional=bi)

        if bi:
            self.classifier = nn.Linear(hidden_num * 2, class_num)
        else:
            # jy: Linear(in_features=107, out_features=29, bias=True)
            self.classifier = nn.Linear(hidden_num, class_num)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, text, label=None):
        """
        text: 维度为 [batch_size, len] 的向量;
        label: 维度为 [batch_size, len] 的向量;
        """
        # jy: 将 token id 进行向量化, 维度为 torch.Size([batch_size, len, embedding_num])
        embedding = self.embedding(text)
        # jy: 经过 lstm 处理后, 得到的 out 维度与 embedding 维度相同;
        out, _ = self.lstm(embedding)
        # jy: 经过处理后, pred 维度为: torch.Size([batch_size, 自己问, 29])
        pred = self.classifier(out)
        self.pred = torch.argmax(pred, dim=-1).reshape(-1)

        if label is not None:
            # jy: pred.shape[-1] 为 class_num
            #     pred.reshape(-1, pred.shape[-1]) 维度为: [batch_size * len, class_num]
            #     label.reshape(-1) 维度为: [batch_size * len]
            loss = self.loss(pred.reshape(-1, pred.shape[-1]), label.reshape(-1))
            return loss
        # jy: torch.argmax(pred, dim=-1) 维度为: [batch_size, len]
        #     torch.argmax(pred, dim=-1).reshape(-1) 后维度为 [batch_size * len]
        return torch.argmax(pred, dim=-1).reshape(-1)

