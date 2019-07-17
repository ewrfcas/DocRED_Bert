import random
import os
import argparse
import numpy as np
import json
import torch
from modeling import BertConfig, BertClsDocRED
from optimization import BERTAdam, warmup_linear
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


def parse_args():
    desc = "BERT_BASELINE"
    _name = 'bert_CLSV0'
    _seed = 556
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--name', type=str, default=_name, help='model name')

    # param
    parser.add_argument('--train_feat_dir', type=str, default='dataset/train_annotated_cls_data.txt',
                        help='training feat dir')
    parser.add_argument('--dev_feat_dir', type=str, default='dataset/dev_cls_data.txt', help='dev feat dir')
    parser.add_argument('--label_num', type=int, default=97)
    parser.add_argument('--checkpoint_dir', type=str, default='check_points', help='checkpoint dir')
    parser.add_argument('--train_batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--context_length', type=int, default=512, help='context_length')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='epoch')
    parser.add_argument('--predict_batch_size', type=int, default=32, help='batch size')
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--seed', type=int, default=_seed, help="random seed for initialization")
    parser.add_argument("--schedule", default='warmup_linear', type=str, help='schedule')
    parser.add_argument("--weight_decay_rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument("--warmup_proportion", default=0.05, type=float,
                        help="Proportion of training to perform linear learning rate warmup")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--bert_config_file",
                        default="/users/caochenjie/EDL_simple/bert_config/wwm_uncased_L-24_H-1024_A-16/bert_config.json",
                        type=str,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument('--fp16', default=True, type=bool)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument("--init_checkpoint",
                        default="/users/caochenjie/EDL_simple/bert_config/wwm_uncased_L-24_H-1024_A-16/pytorch_model.bin",
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


def split_data(features):
    pos_features = []
    neg_features = []
    for feature in features:
        if feature['labels'][0] == 1:
            neg_features.append(feature)
        else:
            pos_features.append(feature)

    return pos_features, neg_features


def combine_shuffle(pos_data, neg_data):
    outputs = []
    for i in range(len(neg_data)):
        outputs.append(torch.cat([pos_data[i], neg_data[i]], 0))
    # shuffle
    rand_index = np.arange(outputs[0].shape[0])
    np.random.shuffle(rand_index)
    for i in range(len(outputs)):
        outputs[i] = outputs[i][rand_index]

    return outputs


def data_gen(data):
    all_input_ids = torch.tensor([f['input_ids'] for f in data], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in data], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in data], dtype=torch.long)
    all_labels = torch.tensor([f['labels'] for f in data], dtype=torch.float)

    return all_input_ids, all_input_mask, all_segment_ids, all_labels


def get_result(y_pred, y_true,
               acc_num, precision_num, recall_num,
               acc_num_ign, precision_num_ign, recall_num_ign, th=0.5):
    intrain = y_true['intrain']
    y_true = y_true['labels']
    y_pred = y_pred[1:]
    y_true = y_true[1:]
    y_pred[y_pred > th] = 1
    y_pred[y_pred < th] = 0

    y_add = y_pred + y_true
    y_add[y_add != 2] = 0
    y_add[y_add == 2] = 1

    recall_num += np.sum(y_true)
    precision_num += np.sum(y_pred)
    acc_num += np.sum(y_add)

    if not intrain:
        recall_num_ign += np.sum(y_true)
        precision_num_ign += np.sum(y_pred)
        acc_num_ign += np.sum(y_add)

    return acc_num, precision_num, recall_num, acc_num_ign, precision_num_ign, recall_num_ign


def evaluate(model, dev_features, device):
    eval_dataloader = DataLoader(dev_features, batch_size=args.predict_batch_size, collate_fn=data_gen,
                                 shuffle=False)

    model.eval()
    eval_index = 0
    acc_num = acc_num_ign = 0
    precision_num = precision_num_ign = 0
    recall_num = recall_num_ign = 0

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
            for i in range(preds.shape[0]):
                acc_num, precision_num, recall_num, \
                acc_num_ign, precision_num_ign, recall_num_ign = get_result(preds[i], dev_features[eval_index],
                                                                            acc_num, precision_num, recall_num,
                                                                            acc_num_ign, precision_num_ign,
                                                                            recall_num_ign)
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

            pbar.set_postfix({'F1': '{:.5f}'.format(f1)})
            pbar.update(1)

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

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    print('loading data...')
    train_features = []
    with open(args.train_feat_dir, 'r') as f:
        for line in tqdm(f):
            train_features.append(json.loads(line))
            # if len(train_features) > 30000:
            #     break
    dev_features = []
    with open(args.dev_feat_dir, 'r') as f:
        for line in tqdm(f):
            dev_features.append(json.loads(line))
            # if len(dev_features) > 3000:
            #     break

    # 分割多数类和少数类数据
    pos_features, neg_features = split_data(train_features)
    # 为了计算学习率渐变，需要求出总训练步数
    num_train_steps = int((len(pos_features) * 2) / args.train_batch_size /
                          args.gradient_accumulation_steps * args.num_train_epochs)

    print("***** Running training *****")
    print("  Num split examples = %d" % len(pos_features) * 2)
    print("  Batch size = %d" % (args.train_batch_size * args.gradient_accumulation_steps))
    print("  Num steps = %d" % num_train_steps)

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

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any([nd in n for nd in no_decay])],
         'weight_decay_rate': args.weight_decay_rate},
        {'params': [p for n, p in param_optimizer if any([nd in n for nd in no_decay])],
         'weight_decay_rate': 0.0}
    ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)

        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BERTAdam(optimizer_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             max_grad_norm=1.0,
                             t_total=num_train_steps,
                             schedule=args.schedule,
                             weight_decay_rate=args.weight_decay_rate)

    train_pos_dataloader = DataLoader(pos_features,
                                      batch_size=args.train_batch_size // 2,
                                      collate_fn=data_gen, shuffle=True)
    neg_features = neg_features[:len(pos_features) * (len(neg_features) // len(pos_features))]
    train_neg_dataloader = DataLoader(neg_features,
                                      batch_size=int(
                                          args.train_batch_size // 2 * (len(neg_features) // len(pos_features))),
                                      collate_fn=data_gen, shuffle=True)
    assert len(train_pos_dataloader) == len(train_neg_dataloader)

    print('start training...')
    model.train()
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        total_loss = 0
        show_dict = set()
        for step, (pos_batch, neg_batch) in enumerate(zip(train_pos_dataloader, train_neg_dataloader)):
            pos_batch = tuple(t.to(device) for t in pos_batch)
            neg_batch = tuple(t[0:args.train_batch_size // 2].to(device) for t in neg_batch)
            batch = combine_shuffle(pos_batch, neg_batch)
            input_ids, input_mask, segment_ids, relation_multi_label = batch
            loss = model(input_ids, segment_ids, input_mask, relation_multi_label)
            total_loss += torch.mean(loss).item()
            if (step + 1) // args.gradient_accumulation_steps % 20 == 0 \
                    and (step + 1) // args.gradient_accumulation_steps not in show_dict:
                print('Epoch = %d/%d steps = %d/%d ' %
                      (epoch + 1, args.num_train_epochs,
                       max(1, (step + 1) // args.gradient_accumulation_steps),
                       num_train_steps // args.num_train_epochs) +
                      'loss = %.6f ' % (total_loss / (step + 1)))
                show_dict.add((step + 1) // args.gradient_accumulation_steps)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used and handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_steps,
                                                                      args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()  # We have accumulated enought gradients
                model.zero_grad()
                global_step += 1

        evaluate(model, dev_features, device)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, 'checkpoint_' + str(global_step) + '.bin')
        torch.save(model_to_save.state_dict(), output_model_file)
