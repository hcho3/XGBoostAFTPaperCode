import pandas as pd
import re
import matplotlib.pyplot as plt

def main():
    db = []

    for data_gen in ['coxph', 'aft']:
        for method in ['SkSurvCoxLinear', 'XGBoostAFTNormal', 'XGBoostAFTLogistic',
                       'XGBoostAFTExtreme', 'XGBoostCox']:
            for censoring_fraction in [0.2, 0.5, 0.8]:
                filename = f'{data_gen}-log/{method}-{censoring_fraction}.txt'
                obj = {'data_gen': data_gen, 'method': method,
                       'censoring_fraction': censoring_fraction}
                with open(filename, 'r') as f:
                    idx = 0
                    for l in f:
                        if idx == 4:
                            m = re.search(r'[0-9\.]+$', l)
                            obj['run_time'] = float(m.group(0))
                        else:
                            m = re.search(r'[0-9\.]+$', l)
                            obj[f'fold_{idx}'] = float(m.group(0))
                            idx += 1
                db.append(obj)
                
    df = pd.json_normalize(db)

    plt.rcParams.update({'font.size': 10.5})
    fig, ax = plt.subplots(4, 3, figsize=(10, 9), sharex='col', sharey='row')

    for i, data_gen in enumerate(['coxph', 'aft']):
        for j, censoring_fraction in enumerate([0.2, 0.5, 0.8]):
            x = df[(df['data_gen'] == data_gen) & (df['censoring_fraction'] == censoring_fraction)]
            x = x.set_index('method')
            ax[i, j].plot(x['fold_0'], 'o', label='Fold 0')
            ax[i, j].plot(x['fold_1'], 'o', label='Fold 1')
            ax[i, j].plot(x['fold_2'], 'o', label='Fold 2')
            ax[i, j].plot(x['fold_3'], 'o', label='Fold 3')
            ax[i, j].tick_params(axis='x', labelrotation=90)
            ax[i, j].set_title(f'data_gen={data_gen}, {censoring_fraction*100:.0f}% censored')
            if j == 0:
                ax[i, j].set_ylabel('Test C-statistics')

    for i, data_gen in enumerate(['coxph', 'aft']):
        for j, censoring_fraction in enumerate([0.2, 0.5, 0.8]):
            x = df[(df['data_gen'] == data_gen) & (df['censoring_fraction'] == censoring_fraction)]
            x = x.set_index('method')
            ax[i+2, j].plot(x['run_time'], 'o', label='Elapsed time')
            ax[i+2, j].tick_params(axis='x', labelrotation=90)
            ax[i+2, j].set_title(f'data_gen={data_gen}, {censoring_fraction*100:.0f}% censored')
            if j == 0:
                ax[i+2, j].set_ylabel('Run time (sec)')

    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor = (0,-0.05,1,1), ncol=4)
    fig.tight_layout()
    fig.savefig('results-plot.pdf', dpi=100, bbox_inches="tight")

if __name__ == '__main__':
    main()
