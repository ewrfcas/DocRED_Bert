import random
import os
import argparse
import numpy as np
import json
import torch
from modeling import BertConfig, BertClsDocRED
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def parse_args():
    desc = "BERT_BASELINE"
    _name = 'bert_CLSV0'
    _seed = 556
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--name', type=str, default=_name, help='model name')

    # param
    parser.add_argument('--dev_feat_dir', type=str, default='dataset/dev_cls_data.txt', help='dev feat dir')
    parser.add_argument('--label_num', type=int, default=97)
    parser.add_argument('--checkpoint_dir', type=str, default='check_points', help='checkpoint dir')
    parser.add_argument('--train_batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--context_length', type=int, default=512, help='context_length')
    parser.add_argument('--predict_batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--ths', type=list, default=[0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    parser.add_argument('--seed', type=int, default=_seed, help="random seed for initialization")
    parser.add_argument("--bert_config_file",
                        default="/users/caochenjie/EDL_simple/bert_config/wwm_uncased_L-24_H-1024_A-16/bert_config.json",
                        type=str,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument('--fp16', default=True, type=bool)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--init_checkpoint",
                        default="check_points/bert_CLSV0/checkpoint_22130.bin",
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--output_dir", type=str, default="check_points/" + _name)

    return check_args(parser.parse_args())


def check_args(args):
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    file_name = os.path.join(args.checkpoint_dir, 'train_setting.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print('------------ Options -------------')
        for k in args.__dict__:
            v = args.__dict__[k]
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print('%s: %s' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
        print('------------ End -------------')

    return args


def data_gen(data):
    all_input_ids = torch.tensor([f['input_ids'] for f in data], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in data], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in data], dtype=torch.long)
    all_labels = torch.tensor([f['labels'] for f in data], dtype=torch.float)

    return all_input_ids, all_input_mask, all_segment_ids, all_labels


def get_result(y_pred, y_true,
               acc_num, precision_num, recall_num,
               acc_num_ign, precision_num_ign, recall_num_ign, th=0.5):
    y_true_tmp = copy.deepcopy(y_true)
    y_pred_tmp = copy.deepcopy(y_pred)

    intrain = y_true_tmp['intrain']
    y_true_tmp = y_true_tmp['labels']

    y_pred_tmp = y_pred_tmp[1:]
    y_true_tmp = y_true_tmp[1:]
    y_pred_tmp[y_pred_tmp > th] = 1
    y_pred_tmp[y_pred_tmp < th] = 0

    y_add = y_pred_tmp + y_true_tmp
    y_add[y_add != 2] = 0
    y_add[y_add == 2] = 1

    recall_num += np.sum(y_true_tmp)
    precision_num += np.sum(y_pred_tmp)
    acc_num += np.sum(y_add)

    if not intrain:
        recall_num_ign += np.sum(y_true_tmp)
        precision_num_ign += np.sum(y_pred_tmp)
        acc_num_ign += np.sum(y_add)

    return acc_num, precision_num, recall_num, acc_num_ign, precision_num_ign, recall_num_ign


def evaluate(model, dev_features, device, ths=[0.5]):
    eval_dataloader = DataLoader(dev_features, batch_size=args.predict_batch_size, collate_fn=data_gen, shuffle=False)

    model.eval()
    all_preds = []

    print("Start evaluating")
    with tqdm(total=len(eval_dataloader), desc='Evaluating') as pbar:
        for batch_data in eval_dataloader:
            input_ids, input_mask, segment_ids, _ = batch_data
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                preds = model(input_ids, segment_ids, input_mask)

            preds = preds.cpu().numpy()
            all_preds.append(preds)
            pbar.update(1)

    for th in ths:
        acc_num = acc_num_ign = 0
        precision_num = precision_num_ign = 0
        recall_num = recall_num_ign = 0
        print('threshold:', th)
        eval_index = 0
        for i in tqdm(range(len(all_preds))):
            for ii in range(all_preds[i].shape[0]):
                acc_num, precision_num, recall_num, \
                acc_num_ign, precision_num_ign, recall_num_ign = get_result(all_preds[i][ii], dev_features[eval_index],
                                                                            acc_num, precision_num, recall_num,
                                                                            acc_num_ign, precision_num_ign,
                                                                            recall_num_ign, th=th)
                eval_index += 1

        recall = acc_num / (recall_num + 1e-5)
        precision = acc_num / (precision_num + 1e-5)
        f1 = 2 * (recall * precision) / (recall + precision + 1e-5)

        recall_ign = acc_num_ign / (recall_num_ign + 1e-5)
        precision_ign = acc_num_ign / (precision_num_ign + 1e-5)
        f1_ign = 2 * (recall_ign * precision_ign) / (recall_ign + precision_ign + 1e-5)

        recall *= 100
        precision *= 100
        f1 *= 100
        recall_ign *= 100
        precision_ign *= 100
        f1_ign *= 100

        print('Precision:{:.3f}, Recall:{:.3f}, F1-score:{:.3f}'.format(precision, recall, f1))
        print('Precision_ignore:{:.3f}, Recall_ignore:{:.3f}, F1-score_ignore:{:.3f}'.format(precision_ign, recall_ign,
                                                                                             f1_ign))

    model.train()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    print("device %s n_gpu %d" % (device, n_gpu))
    print("device: {} n_gpu: {} 16-bits training: {}".format(device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    print('loading data...')
    dev_features = []
    with open(args.dev_feat_dir, 'r') as f:
        for line in tqdm(f):
            dev_features.append(json.loads(line))

    # build the models for training and testing/validation
    print('######## init model ########')
    bert_config = BertConfig.from_json_file(args.bert_config_file)
    model = BertClsDocRED(bert_config, label_num=args.label_num)
    if args.init_checkpoint is not None:
        print('load bert weight...')
        state_dict = torch.load(args.init_checkpoint, map_location='cpu')
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata


        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys,
                                         unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')


        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))

    if args.fp16:
        model.half()

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    evaluate(model, dev_features, device, ths=args.ths)
