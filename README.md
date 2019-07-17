# DocRED_Bert

A bert baseline for DocRED (https://github.com/thunlp/DocRED)

### Difference from the offical baseline (https://github.com/thunlp/DocRED)

1. Use the random undersampling to balance the sample scale of positive and negative (no relation) samples.

2. Only one relation will be contained in one sample.

### Run
1. Download dataset (https://github.com/thunlp/DocRED/tree/master/data)

> Include rel2id.json

2. Download and convert model weights

> Weights in https://github.com/google-research/bert, model and convert in https://github.com/huggingface/pytorch-transformers, 

3. python preprocess.py

> Please use 'convert_feature'.

> Function 'convert_feature_multioutput' is used to build datasets as the offical baseline, but it seems that this strategy works badly in bert.

4. python train_cls.py

### Result in 10 epoch

#### Threshold=0.5 in sigmoid

All: Precision:46.336, Recall:78.334, F1-score:58.228

Ignore: Precision:42.544, Recall:76.520, F1-score:54.684

#### Threshold=0.9 in sigmoid

All: Precision:56.772, Recall:70.718, F1-score:62.981

Ignore: Precision:52.836, Recall:68.814, F1-score:59.776

