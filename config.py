import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="2023 LLM Exam")

    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--over_fitting', type=bool, default=False, help='是否开启过拟合策略')
    parser.add_argument('--gpu', type=str, default='2,3,4,5,6,7')
    parser.add_argument("--train_data_path", type=str, default='data/train.csv', help="训练集")
    parser.add_argument("--test_data_path", type=str, default='data/preliminary_a_test.csv', help="测试集")
    
    parser.add_argument('--pretrain_model_path', type=str, default='./pretrain_models/microsoft_deberta_large')
    # parser.add_argument('--pretrain_model_path', type=str, default='/root/bert_path/longformer-large')

    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--num_epoch', type=int, default=5)

    # base模型的batch size
    parser.add_argument('--train_batch_size', type=int, default=30)
    parser.add_argument('--valid_batch_size', type=int, default=30)
    parser.add_argument('--gradient_step', type=int, default=2)

    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--dropout_ratio', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=5e-6, type=float, help='微调学习率')

    parser.add_argument('--use_EMA', type=bool, default=True)
    parser.add_argument('--ema_decay', type=float, default=0.9)

    parser.add_argument('--use_FGM', type=bool, default=False)
    parser.add_argument('--fgm_eps', type=float, default=0.75)
    
    parser.add_argument('--use_rdrop', type=bool, default=False)

    # 模型保存
    parser.add_argument("--output_file", type=str, default="fintune.csv")
    parser.add_argument("--output_model_path", type=str, default='./save')
    parser.add_argument('--logger_file_path', type=str, default='log/fintune.txt', help="日志输出路径")  # 日志保存路径


    return parser.parse_args()
