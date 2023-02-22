import os.path
from pysenal import read_lines_lazy


"""
中文命名实体识别的标注方法，本文只介绍常见的标注方法。

# 项目目录
命名实体标注
    |-- data
        |-- error.txt       错误标记（word_dict.txt中内容不全）
        |-- Tagging.txt     标记文本
        |-- noTagging.txt   未标记文本
        |-- word_dict.txt   需要标注的内容
    |-- main.py             主函数

# 三种标注方法
BIO（B：begin，I：inside，O：outside）
B-X：实体 X 的开头
I-X：实体的结尾
O：不属于任何类型的

BMES（B：begin，M：middle，E：end，S：single）
B-X：一个词的词首位值
M-X：一个词的中间位置
E-X：一个词的末尾位置
S-X：一个单独的字词


BIOES（B：begin，I：inside，O：outside，E：end，S：single）
B-x：开始
I-x：内部
O：非实体
E-x：实体尾部
S-x：该词本身就是一个实体
"""



def BIO(tag_list, index_start_tag, tag_name, keyword):
    """
    tag_list: 待标注的文本的列表, 长度等于待标注的文本长度;
    index_start_tag: 实体在文本中的起始下标位置;
    tag_name: 实体对应的标签
    keyword: 实体名称;
    """
    for i in range(index_start_tag, index_start_tag + len(keyword)):
        # B
        if index_start_tag == i:
            tag_list[i] = "B-" + tag_name
        # I
        else:
            tag_list[i] = "I-" + tag_name


def BMES(tag_list_BMES, index_start_tag, tag_name, keyword):
    """
    tag_list: 待标注的文本的列表, 长度等于待标注的文本长度;
    index_start_tag: 实体在文本中的起始下标位置;
    tag_name: 实体对应的标签(该方法中不需要使用, 此处传入仅为了做兼容处理)
    keyword: 实体名称;
    """
    len_keyword = len(keyword)
    
    for i in range(index_start_tag, index_start_tag + len_keyword):
        # B
        if index_start_tag == i:
            tag_list_BMES[i] = "B"
        # M
        elif i != index_start_tag + len_keyword - 1:
            tag_list_BMES[i] = "M"
        # E
        elif i == index_start_tag + len_keyword - 1:
            tag_list_BMES[i] = "E"


def BIOES(tag_list, index_start_tag, tag_name, keyword):
    """
    tag_list: 待标注的文本的列表, 长度等于待标注的文本长度;
    index_start_tag: 实体在文本中的起始下标位置;
    tag_name: 实体对应的标签
    keyword: 实体名称;
    """
    len_keyword = len(keyword)
    for i in range(index_start_tag, index_start_tag + len_keyword):
        # S: 一个字符成实体, 标注为: "S-xx"
        if len_keyword == 1:
            tag_list[i] = "S-" + tag_name
        # jy: 以下均为非一个字符成实体;
        # B: 实体首字符标注为: "B-xx"
        elif index_start_tag == i:
            tag_list[i] = "B-" + tag_name
        # I: 实体中间字符标注为: "I-xx"
        elif i != index_start_tag + len_keyword - 1:
            tag_list[i] = "I-" + tag_name
        # E: 实体末尾字符标注为: "E-"
        elif i == index_start_tag + len_keyword - 1:
            tag_list[i] = "E-" + tag_name



def save_tagging(tag_list, f_out, line):
    with open(f_out, 'a', encoding='utf-8') as output_f:
        for w, t in zip(line.strip(), tag_list):
            output_f.write(w + " " + t + '\n')
        output_f.write('\n')


def get_dict_feature_label(f_feature_label, f_error):
    """
    f_feature_label 文件记录了事先拟定好的实体以及其对应的标签, 每一行如: "细菌性疫病 DISEASE"

    返回 dict_feature_label, 实体名称为 key, 实体对应的标签作为 value
    """
    dict_feature_label = {}
    for line in read_lines_lazy(f_feature_label):
        try:
            feature, label = line.split(" ")
            dict_feature_label[feature] = label
        except:
            with open(f_error, 'a', encoding='utf-8') as f:
                f.write(line + "\n")

    return dict_feature_label


def tagging(tagging_method, map_taggingMethodFunc, f_input, f_out, f_feature_label, f_error):
    """
    f_input: 输入文本文件, 每一行为待标注的句子, 如:
             "扁豆细菌性疫病的危害作物是扁豆吗？"
    f_out: 输出文本, 格式如下(多个句子用空格隔开): 
扁 B-CROP
豆 E-CROP
细 B-DISEASE
菌 I-DISEASE
性 I-DISEASE
疫 I-DISEASE
病 E-DISEASE
的 O
危 O
害 O
作 O
物 O
是 O
扁 B-CROP
豆 E-CROP
吗 O
？ O

    进行实体自动标注，用字典中的 key 作为关键词去匹配未标注的文本，将匹配的内容进行标注未 value
    """
    f_out = "%s_%s.txt" % (f_out.rstrip(".txt"), tagging_method)
    # jy: 获取实体名称以及其对应的标签;
    dict_feature_tag = get_dict_feature_label(f_feature_label, f_error)
    # jy: 记录从文本中查找实体时的起始查找下标位置;
    index_log = 0
    # jy: 如果输出文件已存在, 则删除;
    if os.path.exists(f_out):
        os.remove(f_out)

    for line in read_lines_lazy(f_input):
        print("待标注的文本: ", line)

        # O
        len_ = len(line.strip())
        char_other = "S" if tagging_method == "BMES" else "O" 
        tag_list = [char_other for i in range(len_)]

        # jy: 遍历所有实体名称;
        for keyword in dict_feature_tag.keys():
            # jy: 不断循环, 确保文本中对应实体出现多次也都能被标注;
            while True: 
                # jy: 查找实体在文本中的起始下标位置;
                index_start_tag = line.find(keyword, index_log)
                # jy: 如果当前实体不存在于文本中, 则跳出循环
                if index_start_tag == -1:
                    index_log = 0
                    break
                # jy: 更新下一次查找实体的起始下标位置; 只对未标注过的数据进行标注, 防止出现嵌套标注
                index_log = index_start_tag + len(keyword)
                print("文本中查找到的实体以及对应的起始下标: %s: %s" % (keyword, index_start_tag))
                map_taggingMethodFunc[tagging_method](tag_list, index_start_tag, dict_feature_tag[keyword], keyword)
        print(tag_list)
        save_tagging(tag_list, f_out, line)
        print("-"*100)

map_taggingMethodFunc = {
    "BIOES": BIOES,
    "BIO": BIO,
    "BMES": BMES,
}


#tagging_method = "BIOES"
#tagging_method = "BIO"
tagging_method = "BMES"


f_feature_label = "./data/word_dict.txt"
f_error = "./data/error.txt"
f_input = './data/noTagging.txt'
f_output = './data/Tagging.txt'
tagging(tagging_method, map_taggingMethodFunc, f_input, f_output, f_feature_label, f_error)


