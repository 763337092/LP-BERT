import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

DATA_PATH = '../data'

### WN18RR
def WN18RR_concat_data(root_path):
    train = pd.read_csv(f'{root_path}/train.tsv', header=None, sep='\t')
    dev = pd.read_csv(f'{root_path}/dev.tsv', header=None, sep='\t')
    test = pd.read_csv(f'{root_path}/test.tsv', header=None, sep='\t')
    train.columns = dev.columns = test.columns = ['entity1', 'relation', 'entity2']
    print(f'Train|Dev|Test: {len(train)}|{len(dev)}|{len(test)}')

    entity2text = pd.read_csv(f'{root_path}/entity2text.txt', header=None, sep='\t')
    entity2text.columns = ['entity', 'text']
    relation2text = pd.read_csv(f'{root_path}/relation2text.txt', header=None, sep='\t')
    relation2text.columns = ['relation', 'text']
    train = train.merge(entity2text.rename(columns={'entity': 'entity1', 'text': 'text1'}), on='entity1', how='left')
    train = train.merge(relation2text, on='relation', how='left')
    train = train.merge(entity2text.rename(columns={'entity': 'entity2', 'text': 'text2'}), on='entity2', how='left')
    dev = dev.merge(entity2text.rename(columns={'entity': 'entity1', 'text': 'text1'}), on='entity1', how='left')
    dev = dev.merge(relation2text, on='relation', how='left')
    dev = dev.merge(entity2text.rename(columns={'entity': 'entity2', 'text': 'text2'}), on='entity2', how='left')
    test = test.merge(entity2text.rename(columns={'entity': 'entity1', 'text': 'text1'}), on='entity1', how='left')
    test = test.merge(relation2text, on='relation', how='left')
    test = test.merge(entity2text.rename(columns={'entity': 'entity2', 'text': 'text2'}), on='entity2', how='left')

    train.to_csv(f'{root_path}/concat_train.csv', index=False, sep='\t')
    dev.to_csv(f'{root_path}/concat_dev.csv', index=False, sep='\t')
    test.to_csv(f'{root_path}/concat_test.csv', index=False, sep='\t')

print(f'\nWN18RR:')
WN18RR_concat_data(root_path=f'{DATA_PATH}/WN18RR')
print('\numls: ')
WN18RR_concat_data(root_path=f'{DATA_PATH}/umls')
print('\nYAGO3-10: ')
WN18RR_concat_data(root_path=f'{DATA_PATH}/YAGO3-10')

### FB15k-237
def FB15k_concat_data(root_path):
    train = pd.read_csv(f'{root_path}/train.tsv', header=None, sep='\t')
    dev = pd.read_csv(f'{root_path}/dev.tsv', header=None, sep='\t')
    test = pd.read_csv(f'{root_path}/test.tsv', header=None, sep='\t')
    train.columns = dev.columns = test.columns = ['entity1', 'relation', 'entity2']
    print(f'Train|Dev|Test: {len(train)}|{len(dev)}|{len(test)}')

    # entity2text = pd.read_csv(f'{root_path}/fb-entity2text-new.txt', header=None, sep='\t')
    entity2text = pd.read_csv(f'{root_path}/entity2textlong.txt', header=None, sep='\t')
    entity2text.columns = ['entity', 'text']
    relation2text = pd.read_csv(f'{root_path}/relation2text.txt', header=None, sep='\t')
    relation2text.columns = ['relation', 'text']
    train = train.merge(entity2text.rename(columns={'entity': 'entity1', 'text': 'text1'}), on='entity1', how='left')
    train = train.merge(relation2text, on='relation', how='left')
    train = train.merge(entity2text.rename(columns={'entity': 'entity2', 'text': 'text2'}), on='entity2', how='left')
    dev = dev.merge(entity2text.rename(columns={'entity': 'entity1', 'text': 'text1'}), on='entity1', how='left')
    dev = dev.merge(relation2text, on='relation', how='left')
    dev = dev.merge(entity2text.rename(columns={'entity': 'entity2', 'text': 'text2'}), on='entity2', how='left')
    test = test.merge(entity2text.rename(columns={'entity': 'entity1', 'text': 'text1'}), on='entity1', how='left')
    test = test.merge(relation2text, on='relation', how='left')
    test = test.merge(entity2text.rename(columns={'entity': 'entity2', 'text': 'text2'}), on='entity2', how='left')

    train.to_csv(f'{root_path}/concat_train.csv', index=False, sep='\t')
    dev.to_csv(f'{root_path}/concat_dev.csv', index=False, sep='\t')
    test.to_csv(f'{root_path}/concat_test.csv', index=False, sep='\t')

print(f'\nFB15k-237:')
FB15k_concat_data(root_path=f'{DATA_PATH}/FB15k-237')
