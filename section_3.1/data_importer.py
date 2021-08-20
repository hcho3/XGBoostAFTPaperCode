import numpy as np
import pandas as pd
import os

def data_import(current_dir, data_source, data_name):
    data_dir = os.path.join(current_dir, os.path.pardir, 'data', data_name)
    assert os.path.isdir(data_dir)
    if data_source == 'simulated':
        inputFileName = os.path.join(data_dir, 'features.csv')
        labelFileName = os.path.join(data_dir, 'targets.csv')
        foldsFileName = os.path.join(data_dir, 'folds.csv')
        inputs = pd.read_csv(inputFileName).astype(np.double)
        labels = pd.read_csv(labelFileName).astype(np.double)
        folds = pd.read_csv(foldsFileName)
        # pre-processing: change label scale
        labels['min.lambda'] = labels['min.log.penalty'].apply(lambda x: np.exp(float(x)))
        labels['max.lambda'] = labels[' max.log.penalty'].apply(lambda x: np.exp(float(x)))
    elif data_source == 'chipseq':
        inputFileName = os.path.join(data_dir, 'inputs.csv')
        labelFileName = os.path.join(data_dir, 'outputs.csv')
        foldsFileName = os.path.join(data_dir, 'cv/equal_labels/folds.csv')
        inputs = pd.read_csv(inputFileName, index_col='sequenceID').astype(np.double)
        labels = pd.read_csv(labelFileName, index_col='sequenceID').astype(np.double)
        folds = pd.read_csv(foldsFileName, index_col='sequenceID')

        # pre-processing: remove all data columns with INFs and NaNs
        inputs.replace([-float('inf'), float('inf')], np.nan, inplace=True)
        missingCols = inputs.isnull().sum()
        missingCols = list(missingCols[missingCols > 0].index)
        print(f'missingCols = {missingCols}')
        inputs.drop(missingCols, axis=1, inplace=True)
        # pre-processing: remove all data columns with zero variance
        varCols = inputs.apply(lambda x: np.var(x))
        zeroVarCols = list(varCols[varCols == 0].index)
        print(f'zeroVarCols = {zeroVarCols}')
        inputs.drop(zeroVarCols, axis=1, inplace=True)
        # pre-processing: change label scale
        labels['min.lambda'] = labels['min.log.lambda'].apply(lambda x: np.exp(x))
        labels['max.lambda'] = labels['max.log.lambda'].apply(lambda x: np.exp(x))
    else:
        raise ValueError(f'Unknown data source {data_source}')

    return inputs, labels, folds

