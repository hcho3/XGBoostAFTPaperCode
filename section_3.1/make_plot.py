import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def main():
    plt.rcParams.update({'font.size':13})

    fig, axs = plt.subplots(4, 5, figsize=(14, 8.5))
    idx = 0

    ylim = [0, 100]

    datasets = ['ATAC_JV_adipose','CTCF_TDH_ENCODE','H3K27ac-H3K4me3_TDHAM_BP',
                'H3K27ac_TDH_some','H3K36me3_AM_immune','H3K27me3_RL_cancer',
                'H3K27me3_TDH_some','H3K36me3_TDH_ENCODE','H3K36me3_TDH_immune',
                'H3K36me3_TDH_other']

    for dataset_id, dataset in enumerate(datasets):
        ax = axs[idx // 5][idx % 5]
        idx += 1
        nfold = 4
        
        scatter_x = {}
        scatter_y = {}
        for fold in range(1, nfold + 1):
            scatter_x[fold] = []
            scatter_y[fold] = []
        
        for dist_id, dist in enumerate(['normal', 'logistic', 'extreme']):
            for fold in range(1, nfold + 1):
                with open(f'./xgb-aft-log/{dataset}/{dist}-fold{fold}.json', 'r') as f:
                    obj = json.load(f)
                scatter_x[fold].append(dist_id)
                scatter_y[fold].append(obj['final_accuracy'][1] * 100)
                
        with open(f'./penaltyLearning-log/{dataset}/accuracy.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(3)
                scatter_y[fold].append(obj[str(fold)] * 100)
        with open(f'./survreg-log/{dataset}/accuracy.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(4)
                scatter_y[fold].append(obj[str(fold)] * 100)
            
        with open(f'./mmit-log/{dataset}/accuracy.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(5)
                scatter_y[fold].append(obj[str(fold)] * 100)

        for fold in range(1, nfold + 1):
            ax.plot(scatter_x[fold], scatter_y[fold], 'o', label=f'Fold {fold}')
        ax.set_title(f'ChIP-seq ({dataset_id + 1})')
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(['xgb-normal', 'xgb-logistic', 'xgb-extreme', 'intervalCV', 'survReg',
                            'mmit'], rotation='vertical')
        ax.set_ylabel('Test Acc. (%)')
        ax.set_ylim(ylim)
        ax.set_yticks([0, 25, 50, 75, 100])

    ylim = [0, 20]
        
    for dataset_id, dataset in enumerate(datasets):
        ax = axs[idx // 5][idx % 5]
        idx += 1
        
        nfold = 4
        
        scatter_x = {}
        scatter_y = {}
        for fold in range(1, nfold + 1):
            scatter_x[fold] = []
            scatter_y[fold] = []
        
        for dist_id, dist in enumerate(['normal', 'logistic', 'extreme']):
            for fold in range(1, nfold + 1):
                with open(f'./xgb-aft-log/{dataset}/{dist}-fold{fold}.json', 'r') as f:
                    obj = json.load(f)
                scatter_x[fold].append(dist_id)
                scatter_y[fold].append(np.max(obj['timestamp']))
        
        with open(f'./penaltyLearning-log/{dataset}/run_time.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(3)
                scatter_y[fold].append(obj[str(fold)])
        with open(f'./survreg-log/{dataset}/run_time.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(4)
                scatter_y[fold].append(obj[str(fold)])
        with open(f'./mmit-log/{dataset}/run_time.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(5)
                scatter_y[fold].append(obj[str(fold)])
        for fold in range(1, nfold + 1):
            if dataset_id == 2:
                ax.text(x=5, y=0.5, s='(33-48)', fontsize=13, rotation='vertical',
                        horizontalalignment='center', verticalalignment='bottom')
            ax.plot(scatter_x[fold], scatter_y[fold], 'o', label=f'Fold {fold}')
            
        if dataset.startswith('simulated'):
            ax.set_title(dataset)
        else:
            ax.set_title(f'ChIP-seq ({dataset_id + 1})')
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(['xgb-normal', 'xgb-logistic', 'xgb-extreme', 'penaltyLearning',
                            'survReg', 'mmit'], rotation='vertical')
        ax.set_ylabel('Time elapsed (s)')
        ax.set_ylim(ylim)
        ax.set_yticks([0, 5, 10, 15, 20])
        
    for ax in axs.flat:
        ax.label_outer()

    handles, labels = ax.get_legend_handles_labels()
    hl = sorted(zip(handles, labels), key=lambda x : x[1])
    handles, labels = zip(*hl)
    fig.legend(handles, labels, loc='lower center', ncol=4, columnspacing=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 1], w_pad=0.5, h_pad=0)

    plt.savefig('result-plot-ChIPseq.pdf', dpi=150, bbox_inches='tight')

    plt.rcParams.update({'font.size':13})

    fig, axs = plt.subplots(2, 6, figsize=(14, 5.2))
    idx = 0

    ylim = [0, 100]

    for dataset_id, dataset in enumerate(['simulated.abs', 'simulated.linear', 'simulated.sin',
                                          'simulated.model.1', 'simulated.model.2',
                                          'simulated.model.3']):
        ax = axs[idx // 6][idx % 6]
        idx += 1
        nfold = 5
        
        scatter_x = {}
        scatter_y = {}
        for fold in range(1, nfold + 1):
            scatter_x[fold] = []
            scatter_y[fold] = []
        
        for dist_id, dist in enumerate(['normal', 'logistic', 'extreme']):
            for fold in range(1, nfold + 1):
                with open(f'./xgb-aft-log/{dataset}/{dist}-fold{fold}.json', 'r') as f:
                    obj = json.load(f)
                scatter_x[fold].append(dist_id)
                scatter_y[fold].append(obj['final_accuracy'][1] * 100)
                
        with open(f'./penaltyLearning-log/{dataset}/accuracy.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(3)
                scatter_y[fold].append(obj[str(fold)] * 100)
        with open(f'./survreg-log/{dataset}/accuracy.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(4)
                scatter_y[fold].append(obj[str(fold)] * 100)
            
        with open(f'./mmit-log/{dataset}/accuracy.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(5)
                scatter_y[fold].append(obj[str(fold)] * 100)

        for fold in range(1, nfold + 1):
            ax.plot(scatter_x[fold], scatter_y[fold], 'o', label=f'Fold {fold}')
        ax.set_title(dataset)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(['xgb-normal', 'xgb-logistic', 'xgb-extreme', 'intervalCV', 'survReg',
                            'mmit'], rotation='vertical')
        ax.set_ylabel('Test Acc. (%)')
        ax.set_ylim(ylim)
        ax.set_yticks([0, 25, 50, 75, 100])
        
    ylim = [0, 20]

    for dataset_id, dataset in enumerate(['simulated.abs', 'simulated.linear', 'simulated.sin',
                                          'simulated.model.1', 'simulated.model.2',
                                          'simulated.model.3']):
        ax = ax = axs[idx // 6][idx % 6]
        idx += 1
        
        nfold = 5 if dataset.startswith('simulated') else 4
        
        scatter_x = {}
        scatter_y = {}
        for fold in range(1, nfold + 1):
            scatter_x[fold] = []
            scatter_y[fold] = []
        
        for dist_id, dist in enumerate(['normal', 'logistic', 'extreme']):
            for fold in range(1, nfold + 1):
                with open(f'./xgb-aft-log/{dataset}/{dist}-fold{fold}.json', 'r') as f:
                    obj = json.load(f)
                scatter_x[fold].append(dist_id)
                scatter_y[fold].append(np.max(obj['timestamp']))
        
        with open(f'./penaltyLearning-log/{dataset}/run_time.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(3)
                scatter_y[fold].append(obj[str(fold)])
        with open(f'./survreg-log/{dataset}/run_time.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(4)
                scatter_y[fold].append(obj[str(fold)])
        with open(f'./mmit-log/{dataset}/run_time.json', 'r') as f:
            obj = json.load(f)
            for fold in range(1, nfold + 1):
                scatter_x[fold].append(5)
                scatter_y[fold].append(obj[str(fold)])
        for fold in range(1, nfold + 1):
            ax.plot(scatter_x[fold], scatter_y[fold], 'o', label=f'Fold {fold}')
            
        ax.set_title(dataset)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(['xgb-normal', 'xgb-logistic', 'xgb-extreme', 'penaltyLearning',
                            'survReg', 'mmit'], rotation='vertical')
        ax.set_ylabel('Time elapsed (s)')
        ax.set_ylim(ylim)
        ax.set_yticks([0, 5, 10, 15, 20])
        
    for ax in axs.flat:
        ax.label_outer()

    handles, labels = ax.get_legend_handles_labels()
    hl = sorted(zip(handles, labels), key=lambda x : x[1])
    handles, labels = zip(*hl)
    fig.legend(handles, labels, loc='lower center', ncol=5, columnspacing=1.0)
    fig.tight_layout(rect=[0, 0.05, 1, 1], w_pad=0.5, h_pad=0)

    plt.savefig(f'./result-plot-simulated.pdf', dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    main()
