import torch
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from config import parsers


def read_data(f_name):
    """
    _f_name 文件中的数据格式如下:
高 B-NAME
勇 E-NAME
： O
男 O
， O
中 B-CONT
国 M-CONT
国 M-CONT
籍 E-CONT
， O
无 O
境 O
外 O
居 O
留 O
权 O
， O

    返回结果如: 
    ls_all_text: [['寻', '冬', '生', '，'], ['赵', '伟', '先', '生', '，']...]
    ls_all_label: [['B-NAME', 'M-NAME', 'E-NAME', 'O'], ['B-NAME', 'E-NAME', 'O', 'O', 'O']]
    """
    with open(f_name, "r", encoding="utf-8") as f:
        ls_line = f.read().split("\n")

    ls_all_text, ls_all_label, ls_text, ls_label = [], [], [], []
    for data in ls_line:
        if data == "":
            ls_all_text.append(ls_text)
            ls_all_label.append(ls_label)
            ls_text, ls_label = [], []
        else:
            ls_text.append(data.split()[0])
            ls_label.append(data.split()[1])

    # jy: 进行排序, 使得后续遍历每个 batch 的数据时, batch 里的数据长度基本一致;
    ls_tmp = []
    for idx, ls_token in enumerate(ls_all_text):
        ls_tmp.append([len(ls_token), ls_token, ls_all_label[idx]])
    ls_tmp = sorted(ls_tmp, key=lambda x: x[0], reverse=False)
    ls_all_text = [ls_item[1] for ls_item in ls_tmp]
    ls_all_label = [ls_item[2] for ls_item in ls_tmp]


    return ls_all_text, ls_all_label


def build_map(ls_text, ls_text_label):
    """
    构造词表、标签表映射; 返回:
    dict_word_index: {word: index, ...}, 如: {{'唐': 0, '泓': 1, ..., '沾': 1791, '<UNK>': 1792, '<PAD>': 1793}
    dict_label_index: {label: index, ...}, 如: {'B-NAME': 0, 'E-NAME': 1, 'O': 2, 'M-NAME': 3, ...,  'S-ORG': 27, '<PAD>': 28}
    ls_label: 标签类别列表
    """
    dict_word_index, dict_label_index = {}, {}
    ls_label = []
    for text, label in zip(ls_text, ls_text_label):
        for token, token_label in zip(text, label):
            if token not in dict_word_index:
                dict_word_index[token] = len(dict_word_index)
            if token_label not in dict_label_index:
                dict_label_index[token_label] = len(dict_label_index)
                ls_label.append(token_label)
    dict_word_index['<UNK>'] = len(dict_word_index)
    dict_word_index['<PAD>'] = len(dict_word_index)
    dict_label_index['<PAD>'] = len(dict_label_index)
    ls_label.append('<PAD>')
    return dict_word_index, dict_label_index, ls_label


class BiLSTMDataset(Dataset):
    def __init__(self, args, ls_text, ls_label, dict_word_index, dict_label_index):
        self.ls_text = ls_text
        self.ls_label = ls_label
        self.dict_word_index = dict_word_index
        self.dict_label_index = dict_label_index
        self.args = args

    def __getitem__(self, index):
        text = self.ls_text[index]
        label = self.ls_label[index]

        text_id = [self.dict_word_index.get(i, self.dict_word_index["<UNK>"]) for i in text]
        label_id = [self.dict_label_index[i] for i in label]

        return text_id, label_id

    def __len__(self):
        return self.ls_text.__len__()

    def pro_batch_data(self, batch_data):
        texts, labels, batch_len = [], [], []
        for i in batch_data:
            texts.append(i[0])
            labels.append(i[1])
            batch_len.append(len(i[0]))

        max_batch_len = max(batch_len)

        texts = [i + [self.dict_word_index["<PAD>"]] * (max_batch_len - len(i)) for i in texts]
        labels = [i + [self.dict_label_index["<PAD>"]] * (max_batch_len - len(i)) for i in labels]

        texts = torch.tensor(texts, dtype=torch.int64, device=self.args.device if torch.cuda.is_available() else "cpu")
        labels = torch.tensor(labels, dtype=torch.long, device=self.args.device if torch.cuda.is_available() else "cpu")

        return texts, labels


def prepare_data():
    args = parsers()
    ls_train_text, ls_train_label = read_data(args.train_file)
    ls_dev_text, ls_dev_label = read_data(args.dev_file)
    ls_test_text, ls_test_label = read_data(args.test_file)
    # jy: 返回结果如下:
    """
    dict_word_index: {word: index, ...}, 如: {{'唐': 0, '泓': 1, ..., '沾': 1791, '<UNK>': 1792, '<PAD>': 1793}
    dict_label_index: {label: index, ...}, 如: {'B-NAME': 0, 'E-NAME': 1, 'O': 2, 'M-NAME': 3, ...,  'S-ORG': 27, '<PAD>': 28}
    ls_label: 标签类别列表(index 不固定, 并不一定与 label_index 中的相同)
    """
    word_index, label_index, index_label = build_map(ls_train_text, ls_train_label)

    # 所有不重复的汉字
    token_num = len(word_index)
    # 所有类别
    class_num = len(label_index)

    train_dataset = BiLSTMDataset(args, ls_train_text, ls_train_label, word_index, label_index)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=train_dataset.pro_batch_data)

    dev_dataset = BiLSTMDataset(args, ls_dev_text, ls_dev_label, word_index, label_index)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=train_dataset.pro_batch_data)

    test_dataset = BiLSTMDataset(args, ls_test_text, ls_test_label, word_index, label_index)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=train_dataset.pro_batch_data)

    pkl.dump([word_index, label_index, index_label, token_num, class_num], open(args.data_pkl, "wb"))

    return train_dataloader, dev_dataloader, test_dataloader, index_label, token_num, class_num


