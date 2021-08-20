
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--distribution', required=True, choices=['normal', 'logistic', 'extreme'])
    parser.add_argument('--test_fold_id', required=True, type=int, choices=[1, 2, 3, 4])
    parser.add_argument('--seed', required=False, type=int, default=1)
    parser.add_argument('--nthread', required=True, type=int)
    parser.add_argument('--s3_bucket', required=True)
    parser.add_argument('--hyperparameters', required=True, nargs='+')

    args = parser.parse_args()

    cmb = '-'.join(args.hyperparameters)

    print(f'{args.run}-{cmb}-{args.dataset}-{args.test_fold_id}-{args.distribution}.txt')

if __name__ == '__main__':
    main()
