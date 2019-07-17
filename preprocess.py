from tokenization import FullTokenizer
from tqdm import tqdm
import json
import copy

tokenizer = FullTokenizer('/users/caochenjie/EDL_simple/bert_config/wwm_uncased_L-24_H-1024_A-16/vocab.txt',
                          do_lower_case=True)
rel2id = json.load(open('dataset/rel2id.json', 'r'))
fact_in_train = set()
span_wrong_dict = set()


def convert_feature(file_name, output_file, max_seq_length=512, is_training=True, is_test=False):
    i_line = 0
    max_len_for_doc = max_seq_length - 2  # [CLS] [SEP]

    pos_samples = 0
    neg_samples = 0

    print('convert features...')
    with open(output_file, 'w') as w:
        with open(file_name, 'r') as f:
            data_samples = json.load(f)
            for sample in tqdm(data_samples):

                if not is_test:
                    labels = sample['labels']
                # 外面先wordpiece分词,映射每句的word index
                sents = []
                sent_map = []
                for sent in sample['sents']:
                    new_sent = []
                    new_map = {}
                    for i_t, token in enumerate(sent):
                        tokens_wordpiece = tokenizer.tokenize(token)
                        new_map[i_t] = len(new_sent)
                        new_sent.extend(tokens_wordpiece)
                    new_map[i_t + 1] = len(new_sent)
                    sent_map.append(new_map)
                    sents.append(new_sent)

                entitys = sample['vertexSet']

                # 先存储一波有relation的实体关系
                train_triple = {}
                if not is_test:
                    for label in labels:
                        evidence = label['evidence']
                        r = int(rel2id[label['r']])
                        # 由于同一组实体可能存在多个关系，这里要用list存！
                        if (label['h'], label['t']) not in train_triple:
                            train_triple[(label['h'], label['t'])] = [{'relation': r, 'evidence': evidence}]
                        else:  # 不过要确保他们的关系是不同的
                            in_triple = False
                            for tmp_r in train_triple[(label['h'], label['t'])]:
                                if tmp_r['relation'] == r:
                                    in_triple = True
                                    break
                            if not in_triple:
                                train_triple[(label['h'], label['t'])].append({'relation': r, 'evidence': evidence})

                        intrain = False
                        # 登记哪些实体关系在train中出现过了
                        for e1i in entitys[label['h']]:
                            for e2i in entitys[label['t']]:
                                if is_training:
                                    fact_in_train.add((e1i['name'], e2i['name'], r))
                                elif not is_test:
                                    # 验证集查找
                                    if (e1i['name'], e2i['name'], r) in fact_in_train:
                                        for train_tmp in train_triple[(label['h'], label['t'])]:
                                            train_tmp['intrain'] = True
                                        intrain = True
                        if not intrain:
                            for train_tmp in train_triple[(label['h'], label['t'])]:
                                train_tmp['intrain'] = False

                # 遍历所有实体构建关系，没有关系的打上NA
                for e1, entity1 in enumerate(entitys):
                    for e2, entity2 in enumerate(entitys):
                        if e1 != e2:
                            # 在$所有$实体1前后加上[unused0]和[unused1]用来给实体定位,在$所有$实体2前后加上[unused2]和[unused3]用来给实体定位
                            # [unused0] Hirabai Badodekar [unused1] , Gangubai Hangal , Mogubai Kurdikar ) ,
                            # made the [unused2] Indian [unused3] classical music so much greater .

                            entity1_ = copy.deepcopy(entity1)
                            entity2_ = copy.deepcopy(entity2)
                            for e in entity1_:
                                e['first'] = True  # 是entity1
                            for e in entity2_:
                                e['first'] = False  # 是entity2
                            new_sents = copy.deepcopy(sents)
                            # 吧entity按照pos从后往前排序，起点相同根据终点倒序排, 这样insert可以无视pos的offset
                            sorted_entity = sorted(entity1_ + entity2_, key=lambda x: (x['pos'][0], x['pos'][1]),
                                                   reverse=True)

                            start_end_dict = set()  # 为了记录起点和终点是否有重叠
                            for se in sorted_entity:
                                map_start = sent_map[se['sent_id']][se['pos'][0]]
                                map_end = sent_map[se['sent_id']][se['pos'][1]]

                                # 如果有重叠，起点+1，终点+2，否则起点不变，终点+1(因为起点+了一个标识)
                                if (map_start, se['sent_id']) in start_end_dict:
                                    map_start_fin = map_start + 1
                                    map_end_fin = map_end + 2
                                else:
                                    map_start_fin = map_start
                                    map_end_fin = map_end + 1
                                    start_end_dict.add((map_start, se['sent_id']))

                                # entity_span = ' '.join(new_sents[se['sent_id']][map_start_fin:map_end_fin - 1])
                                # entity_span = entity_span.replace(' ,', ',').replace(' .', '.'). \
                                #     replace(' \'', '\'').replace(' :', ':').replace(' ##', '').replace('##', ''). \
                                #     replace(' :', ':')

                                # # 确定存在重叠
                                # if se['name'].lower() != entity_span and '[unused' not in entity_span \
                                #         and entity_span not in span_wrong_dict:
                                #     # 检查entity span
                                #     span_wrong_dict.add(entity_span)
                                #     print('entity span wrong:', se['name'].lower(), 'V.S.', entity_span)

                                # 由于存在实体重叠的情况，这里我们是按照起点倒序排的，起点相同根据终点倒序排
                                # 重叠的时候start+1, end需要+2
                                # 所以最后可能出现[unused0] tokens, tokens, tokens[unused1]
                                # -> [unused0] [unused2] tokens tokens [unused3] tokens [unused1]
                                # 同时实体可能完全重叠，此刻也不会有问题
                                # 所以最后可能出现[unused0]tokens, tokens, tokens[unused1]
                                # -> [unused0] [unused2] tokens tokens tokens [unused3] [unused1]
                                if se['first']:  # 混合排序后区分entity1和entity2
                                    new_sents[se['sent_id']].insert(map_start_fin, '[unused0]')
                                    new_sents[se['sent_id']].insert(map_end_fin, '[unused1]')
                                else:
                                    new_sents[se['sent_id']].insert(map_start_fin, '[unused2]')
                                    new_sents[se['sent_id']].insert(map_end_fin, '[unused3]')

                            doc_tokens = []
                            for sent in new_sents:
                                doc_tokens.extend(sent)

                            if len(doc_tokens) > max_len_for_doc:
                                continue
                                # TODO doc_tokens = doc_tokens[:max_len_for_doc]

                            tokens = ['[CLS]'] + doc_tokens + ['[SEP]']
                            segment_ids = [0] * (len(doc_tokens) + 2)
                            input_ids = tokenizer.convert_tokens_to_ids(tokens)
                            input_mask = [1] * len(input_ids)

                            intrain = None
                            relation_label = None
                            evidence = []
                            if not is_test:
                                if (e1, e2) not in train_triple:
                                    relation_label = [0] * len(rel2id)
                                    relation_label[0] = 1
                                    evidence = []
                                    intrain = False
                                    neg_samples += 1
                                else:
                                    relation_label = [0] * len(rel2id)
                                    # 一个实体可能存在多个关系
                                    for train_tmp in train_triple[(e1, e2)]:
                                        relation_label[train_tmp['relation']] = 1
                                        evidence.append(train_tmp['evidence'])
                                    intrain = train_triple[(e1, e2)][0]['intrain']
                                    pos_samples += 1

                            # Zero-pad up to the sequence length.
                            while len(input_ids) < max_seq_length:
                                input_ids.append(0)
                                input_mask.append(0)
                                segment_ids.append(0)

                            assert len(input_ids) == max_seq_length
                            assert len(input_mask) == max_seq_length
                            assert len(segment_ids) == max_seq_length

                            if i_line <= 5:
                                print('#' * 100)
                                print('E1:', [e['name'] for e in entity1])
                                print('E2:', [e['name'] for e in entity2])
                                print('intrain:', intrain)
                                print('Evidence:', evidence)
                                print('tokens:', tokens)
                                print('segment ids:', segment_ids)
                                print('input ids:', input_ids)
                                print('input mask', input_mask)
                                print('relation_label:', relation_label)

                            i_line += 1

                            feature = {'input_ids': input_ids,
                                       'input_mask': input_mask,
                                       'segment_ids': segment_ids,
                                       'labels': relation_label,
                                       'evidences': evidence,
                                       'intrain': intrain}

                            w.write(json.dumps(feature, ensure_ascii=False) + '\n')

    print(output_file, 'final samples', i_line)
    print('pos samples:', pos_samples)
    print('neg samples:', neg_samples)


# def convert_feature_multioutput(file_name, output_file, max_seq_length=512, is_training=True, is_test=False):
#     i_line = 0
#     max_len_for_doc = max_seq_length - 2  # [CLS] [SEP]
#
#     pos_samples = 0
#     neg_samples = 0
#
#     print('convert features...')
#     with open(output_file, 'w') as w:
#         with open(file_name, 'r') as f:
#             data_samples = json.load(f)
#             for sample in tqdm(data_samples):
#
#                 if not is_test:
#                     labels = sample['labels']
#                 # 外面先wordpiece分词,映射每句的word index
#                 sents = []
#                 sent_map = []
#                 sent_lengths = []
#                 L = 1  # [CLS]是第一个
#                 for sent in sample['sents']:
#                     new_sent = []
#                     new_map = {}
#                     sent_lengths.append(L)
#                     for i_t, token in enumerate(sent):
#                         tokens_wordpiece = tokenizer.tokenize(token)
#                         new_map[i_t] = len(new_sent)
#                         new_sent.extend(tokens_wordpiece)
#                     new_map[i_t + 1] = len(new_sent)
#                     sent_map.append(new_map)
#                     sents.append(new_sent)
#                     L += len(new_sent)
#
#                 entitys = sample['vertexSet']
#
#                 # 先存储一波有relation的实体关系
#                 train_triple = {}
#                 if not is_test:
#                     for label in labels:
#                         evidence = label['evidence']
#                         r = int(rel2id[label['r']])
#                         # 由于同一组实体可能存在多个关系，这里要用list存！
#                         if (label['h'], label['t']) not in train_triple:
#                             train_triple[(label['h'], label['t'])] = [{'relation': r, 'evidence': evidence}]
#                         else:  # 不过要确保他们的关系是不同的
#                             in_triple = False
#                             for tmp_r in train_triple[(label['h'], label['t'])]:
#                                 if tmp_r['relation'] == r:
#                                     in_triple = True
#                                     break
#                             if not in_triple:
#                                 train_triple[(label['h'], label['t'])].append({'relation': r, 'evidence': evidence})
#
#                         intrain = False
#                         # 登记哪些实体关系在train中出现过了
#                         for e1i in entitys[label['h']]:
#                             for e2i in entitys[label['t']]:
#                                 if is_training:
#                                     fact_in_train.add((e1i['name'], e2i['name'], r))
#                                 elif not is_test:
#                                     # 验证集查找
#                                     if (e1i['name'], e2i['name'], r) in fact_in_train:
#                                         for train_tmp in train_triple[(label['h'], label['t'])]:
#                                             train_tmp['intrain'] = True
#                                         intrain = True
#                         if not intrain:
#                             for train_tmp in train_triple[(label['h'], label['t'])]:
#                                 train_tmp['intrain'] = False
#
#                 doc_tokens = []
#                 for sent in sents:
#                     doc_tokens.extend(sent)
#
#                 if len(doc_tokens) > max_len_for_doc:
#                     continue
#                     # TODO doc_tokens = doc_tokens[:max_len_for_doc]
#
#                 tokens = ['[CLS]'] + doc_tokens + ['[SEP]']
#                 segment_ids = [0] * (len(doc_tokens) + 2)
#                 input_ids = tokenizer.convert_tokens_to_ids(tokens)
#                 input_mask = [1] * len(input_ids)
#
#                 # Zero-pad up to the sequence length.
#                 while len(input_ids) < max_seq_length:
#                     input_ids.append(0)
#                     input_mask.append(0)
#                     segment_ids.append(0)
#
#                 assert len(input_ids) == max_seq_length
#                 assert len(input_mask) == max_seq_length
#                 assert len(segment_ids) == max_seq_length
#
#                 entity_list_pos = []
#                 entity_list_neg = []
#
#                 # 遍历所有实体构建关系，没有关系的打上NA
#                 for e1, entity1 in enumerate(entitys):
#                     for e2, entity2 in enumerate(entitys):
#                         if e1 != e2:
#                             h_starts = []
#                             h_ends = []
#                             for entity1_ in entity1:
#                                 h_starts.append(sent_map[entity1_['sent_id']][entity1_['pos'][0]] +
#                                                 sent_lengths[entity1_['sent_id']])
#                                 h_ends.append(sent_map[entity1_['sent_id']][entity1_['pos'][1]] +
#                                               sent_lengths[entity1_['sent_id']])
#
#                                 # ########### test code ############
#                                 # h_span = ' '.join(tokens[h_starts[-1]:h_ends[-1]])
#                                 # h_span = h_span.replace(' ,', ',').replace(' .', '.'). \
#                                 #     replace(' \'', '\'').replace(' :', ':').replace(' ##', '').replace('##', ''). \
#                                 #     replace(' :', ':')
#                                 #
#                                 # if entity1_['name'].lower() != h_span and h_span not in span_wrong_dict:
#                                 #     # 检查entity span
#                                 #     span_wrong_dict.add(h_span)
#                                 #     print('entity span wrong:', entity1_['name'].lower(), 'V.S.', h_span)
#
#                             t_starts = []
#                             t_ends = []
#                             for entity2_ in entity2:
#                                 t_starts.append(sent_map[entity2_['sent_id']][entity2_['pos'][0]] +
#                                                 sent_lengths[entity2_['sent_id']])
#                                 t_ends.append(sent_map[entity2_['sent_id']][entity2_['pos'][1]] +
#                                               sent_lengths[entity2_['sent_id']])
#
#                                 # ########### test code ############
#                                 # h_span = ' '.join(tokens[t_starts[-1]:t_ends[-1]])
#                                 # h_span = h_span.replace(' ,', ',').replace(' .', '.'). \
#                                 #     replace(' \'', '\'').replace(' :', ':').replace(' ##', '').replace('##', ''). \
#                                 #     replace(' :', ':')
#                                 #
#                                 # if entity2_['name'].lower() != h_span and h_span not in span_wrong_dict:
#                                 #     # 检查entity span
#                                 #     span_wrong_dict.add(h_span)
#                                 #     print('entity span wrong:', entity2_['name'].lower(), 'V.S.', h_span)
#
#                             evidence = []
#                             if not is_test:
#                                 if (e1, e2) not in train_triple:
#                                     relation_label = [0]
#                                     evidence = []
#                                     intrain = False
#                                     neg_samples += 1
#                                     entity_list_neg.append({'h_starts': h_starts,
#                                                             'h_ends': h_ends,
#                                                             't_starts': t_starts,
#                                                             't_ends': t_ends,
#                                                             'intrain': intrain,
#                                                             'evidence': evidence,
#                                                             'relation_label': relation_label})
#                                 else:
#                                     relation_label = []
#                                     # 一个实体可能存在多个关系
#                                     for train_tmp in train_triple[(e1, e2)]:
#                                         relation_label.append(train_tmp['relation'])
#                                         evidence.append(train_tmp['evidence'])
#                                     intrain = train_triple[(e1, e2)][0]['intrain']
#                                     pos_samples += 1
#                                     entity_list_pos.append({'h_starts': h_starts,
#                                                             'h_ends': h_ends,
#                                                             't_starts': t_starts,
#                                                             't_ends': t_ends,
#                                                             'intrain': intrain,
#                                                             'evidence': evidence,
#                                                             'relation_label': relation_label})
#                 if len(entity_list_pos) == 0 and is_training:
#                     continue
#
#                 i_line += 1
#
#                 if is_training:
#                     feature = {'input_ids': input_ids,
#                                'input_mask': input_mask,
#                                'segment_ids': segment_ids,
#                                'entity_list_pos': entity_list_pos,
#                                'entity_list_neg': entity_list_neg}
#                 else:
#                     feature = {'input_ids': input_ids,
#                                'input_mask': input_mask,
#                                'segment_ids': segment_ids,
#                                'entity_list': entity_list_pos + entity_list_neg}
#
#                 w.write(json.dumps(feature, ensure_ascii=False) + '\n')
#
#     print(output_file, 'final samples', i_line)
#     print('pos samples:', pos_samples)
#     print('neg samples:', neg_samples)


file_list = ['dataset/train_annotated.json', 'dataset/dev.json', 'dataset/test.json']
for file_name in file_list:
    # output_file = file_name.split('/')[0] + '/' + file_name.split('/')[-1].split('.json')[0] + '_data.txt'
    # convert_feature_multioutput(file_name, output_file, is_training=True if 'train' in file_name else False,
    #                             is_test=True if 'test' in file_name else False)

    output_file = file_name.split('/')[0] + '/' + file_name.split('/')[-1].split('.json')[0] + '_cls_data.txt'
    convert_feature(file_name, output_file, is_training=True if 'train' in file_name else False,
                    is_test=True if 'test' in file_name else False)
