import os
import argparse


def parsers():

    device_num = 0
    dir_model_out = "./model_out"
    dir_data = "./data"  

    parser = argparse.ArgumentParser(description="BiLSTM entity recognition of argparse")
    parser.add_argument('--train_file', type=str, default=os.path.join("data", "train.char.bmes"))
    parser.add_argument('--test_file', type=str, default=os.path.join("data", "test.char.bmes"))
    parser.add_argument('--dev_file', type=str, default=os.path.join("data", "dev.char.bmes"))
    parser.add_argument('--data_pkl', type=str, default=os.path.join("data", "data_parameter.pkl"))

    # jy: GPU 编号, 如果使用 cpu, 则对应的值设置为 "cpu"
    parser.add_argument("--device", type=str, default="cuda:%d" % device_num)
    # jy: 是否仅跑测试结果, 不进行训练;
    #parser.add_argument("--is_do_test_only", type=bool, default=False)
    parser.add_argument("--is_do_test_only", type=bool, default=True)
    # jy: 是否仅进行 predict, 不进行训练;
    parser.add_argument("--is_do_predict_only", type=bool, default=False)
    #parser.add_argument("--is_do_predict_only", type=bool, default=True)

    parser.add_argument('--epochs', type=int, default=230)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_num', type=int, default=101)
    parser.add_argument('--hidden_num', type=int, default=107)
    parser.add_argument('--learn_rate', type=float, default=1e-4)
    parser.add_argument('--bi', type=bool, default=False)
    parser.add_argument("--save_model_best", type=str, default=os.path.join(dir_model_out, "best_model.pth"))
    parser.add_argument("--save_model_last", type=str, default=os.path.join(dir_model_out, "last_model.pth"))
    args = parser.parse_args()
    return args
