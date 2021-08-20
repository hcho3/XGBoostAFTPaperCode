job_list = [
    ('H3K27ac-H3K4me3_TDHAM_BP', 'normal', 1),
    ('H3K27ac-H3K4me3_TDHAM_BP', 'normal', 2),
    ('H3K27ac-H3K4me3_TDHAM_BP', 'normal', 3),
    ('H3K27ac-H3K4me3_TDHAM_BP', 'normal', 4),
    ('H3K27ac-H3K4me3_TDHAM_BP', 'logistic', 1),
    ('H3K27ac-H3K4me3_TDHAM_BP', 'logistic', 2),
    ('H3K27ac-H3K4me3_TDHAM_BP', 'logistic', 3),
    ('H3K27ac-H3K4me3_TDHAM_BP', 'logistic', 4),
    ('H3K27ac-H3K4me3_TDHAM_BP', 'extreme', 1),
    ('H3K27ac-H3K4me3_TDHAM_BP', 'extreme', 2),
    ('H3K27ac-H3K4me3_TDHAM_BP', 'extreme', 3),
    ('H3K27ac-H3K4me3_TDHAM_BP', 'extreme', 4),
    ('H3K27ac_TDH_some', 'normal', 1),
    ('H3K27ac_TDH_some', 'normal', 2),
    ('H3K27ac_TDH_some', 'normal', 3),
    ('H3K27ac_TDH_some', 'normal', 4),
    ('H3K27ac_TDH_some', 'logistic', 1),
    ('H3K27ac_TDH_some', 'logistic', 2),
    ('H3K27ac_TDH_some', 'logistic', 3),
    ('H3K27ac_TDH_some', 'logistic', 4),
    ('H3K27ac_TDH_some', 'extreme', 1),
    ('H3K27ac_TDH_some', 'extreme', 2),
    ('H3K27ac_TDH_some', 'extreme', 3),
    ('H3K27ac_TDH_some', 'extreme', 4),
    ('H3K36me3_AM_immune', 'normal', 1),
    ('H3K36me3_AM_immune', 'normal', 2),
    ('H3K36me3_AM_immune', 'normal', 3),
    ('H3K36me3_AM_immune', 'normal', 4),
    ('H3K36me3_AM_immune', 'logistic', 1),
    ('H3K36me3_AM_immune', 'logistic', 2),
    ('H3K36me3_AM_immune', 'logistic', 3),
    ('H3K36me3_AM_immune', 'logistic', 4),
    ('H3K36me3_AM_immune', 'extreme', 1),
    ('H3K36me3_AM_immune', 'extreme', 2),
    ('H3K36me3_AM_immune', 'extreme', 3),
    ('H3K36me3_AM_immune', 'extreme', 4),
    ('H3K27me3_RL_cancer', 'normal', 1),
    ('H3K27me3_RL_cancer', 'normal', 2),
    ('H3K27me3_RL_cancer', 'normal', 3),
    ('H3K27me3_RL_cancer', 'normal', 4),
    ('H3K27me3_RL_cancer', 'logistic', 1),
    ('H3K27me3_RL_cancer', 'logistic', 2),
    ('H3K27me3_RL_cancer', 'logistic', 3),
    ('H3K27me3_RL_cancer', 'logistic', 4),
    ('H3K27me3_RL_cancer', 'extreme', 1),
    ('H3K27me3_RL_cancer', 'extreme', 2),
    ('H3K27me3_RL_cancer', 'extreme', 3),
    ('H3K27me3_RL_cancer', 'extreme', 4),
    ('H3K27me3_TDH_some', 'normal', 1),
    ('H3K27me3_TDH_some', 'normal', 2),
    ('H3K27me3_TDH_some', 'normal', 3),
    ('H3K27me3_TDH_some', 'normal', 4),
    ('H3K27me3_TDH_some', 'logistic', 1),
    ('H3K27me3_TDH_some', 'logistic', 2),
    ('H3K27me3_TDH_some', 'logistic', 3),
    ('H3K27me3_TDH_some', 'logistic', 4),
    ('H3K27me3_TDH_some', 'extreme', 1),
    ('H3K27me3_TDH_some', 'extreme', 2),
    ('H3K27me3_TDH_some', 'extreme', 3),
    ('H3K27me3_TDH_some', 'extreme', 4),
    ('H3K36me3_TDH_ENCODE', 'normal', 1),
    ('H3K36me3_TDH_ENCODE', 'normal', 2),
    ('H3K36me3_TDH_ENCODE', 'normal', 3),
    ('H3K36me3_TDH_ENCODE', 'normal', 4),
    ('H3K36me3_TDH_ENCODE', 'logistic', 1),
    ('H3K36me3_TDH_ENCODE', 'logistic', 2),
    ('H3K36me3_TDH_ENCODE', 'logistic', 3),
    ('H3K36me3_TDH_ENCODE', 'logistic', 4),
    ('H3K36me3_TDH_ENCODE', 'extreme', 1),
    ('H3K36me3_TDH_ENCODE', 'extreme', 2),
    ('H3K36me3_TDH_ENCODE', 'extreme', 3),
    ('H3K36me3_TDH_ENCODE', 'extreme', 4),
    ('H3K36me3_TDH_immune', 'normal', 1),
    ('H3K36me3_TDH_immune', 'normal', 2),
    ('H3K36me3_TDH_immune', 'normal', 3),
    ('H3K36me3_TDH_immune', 'normal', 4),
    ('H3K36me3_TDH_immune', 'logistic', 1),
    ('H3K36me3_TDH_immune', 'logistic', 2),
    ('H3K36me3_TDH_immune', 'logistic', 3),
    ('H3K36me3_TDH_immune', 'logistic', 4),
    ('H3K36me3_TDH_immune', 'extreme', 1),
    ('H3K36me3_TDH_immune', 'extreme', 2),
    ('H3K36me3_TDH_immune', 'extreme', 3),
    ('H3K36me3_TDH_immune', 'extreme', 4),
    ('H3K36me3_TDH_other', 'normal', 1),
    ('H3K36me3_TDH_other', 'normal', 2),
    ('H3K36me3_TDH_other', 'normal', 3),
    ('H3K36me3_TDH_other', 'normal', 4),
    ('H3K36me3_TDH_other', 'logistic', 1),
    ('H3K36me3_TDH_other', 'logistic', 2),
    ('H3K36me3_TDH_other', 'logistic', 3),
    ('H3K36me3_TDH_other', 'logistic', 4),
    ('H3K36me3_TDH_other', 'extreme', 1),
    ('H3K36me3_TDH_other', 'extreme', 2),
    ('H3K36me3_TDH_other', 'extreme', 3),
    ('H3K36me3_TDH_other', 'extreme', 4),
]
print(len(job_list))

import io
import os
import boto3
import time
import itertools

import asyncio
import asyncssh

ecr_repo = '812345574397.dkr.ecr.us-west-2.amazonaws.com'
s3_bucket = 'aft-experiment-logs'

cmd_template = (
   '{{ /home/ubuntu/miniconda3/bin/python -m awscli ecr get-login-password --region us-west-2 ' +
   '| docker login --username AWS --password-stdin {ecr_repo} ; }} && ' +
   'docker pull {ecr_repo}/aft_exp:latest && ' +
   'docker run --rm -it {ecr_repo}/aft_exp:latest --dataset {dataset} ' +
   '--distribution {distribution} --test_fold_id {test_fold_id} --seed 1 --sampler {sampler} ' +
   '--nthread {nthread} --ntrial {ntrial} --s3_bucket {s3_bucket}'
)

opts = asyncssh.SSHClientConnectionOptions(
    client_keys='./bench.pem',
    username='ubuntu',
    password=None,  #  Do not permit password authentication
    known_hosts=None
)

async def run_client(hostname, cmd, max_tries, logfile):
    sleep_time = 5
    with open(logfile, 'w') as f:
        print('', file=f, end='')
    for _ in range(max_tries):
        try:
            async with asyncssh.connect(hostname, options=opts) as conn:
                print(f'[{hostname}] Conneciton successful')
                async with conn.create_process(cmd, term_type='xterm-color',
                                               stderr=asyncssh.STDOUT) as process:
                    async for line in process.stdout:
                        with open(logfile, 'a') as f:
                            print(line, file=f, end='')
                    await process.wait(check=True)
                break
        except ConnectionRefusedError as e:
            print(f'[{hostname}] Connection failed: {str(e)}. Will retry in {sleep_time} sec')
            await asyncio.sleep(sleep_time)
            sleep_time *= 2  # Exponential backoff

def log_name(e):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{e[0]}-{e[1]}-test{e[2]}.txt')

async def main_loop(instances, job_list, max_tries):
    tasks = [run_client(
                x.public_dns_name,
                cmd_template.format(
                    ecr_repo=ecr_repo,
                    dataset=job_list[i][0],
                    distribution=job_list[i][1],
                    test_fold_id=job_list[i][2],
                    sampler='random',
                    nthread=48,
                    ntrial=1000,
                    s3_bucket=s3_bucket
                ),
                max_tries,
                logfile=log_name(job_list[i]))
             for i, x in enumerate(instances)]
    await asyncio.gather(*tasks)

def main():
    nworker = 24

    for i in range(nworker):
        print(f'Worker {i} gets job {job_list[i]}')

    ec2 = boto3.resource('ec2', region_name='us-west-2')
    instances = ec2.create_instances(
            ImageId='ami-0f652a9c7bb352e2e',
            InstanceType='c5.24xlarge',
            KeyName='bench',
            SecurityGroups=['launch-wizard-4'],
            IamInstanceProfile={
                'Name': 'AFTExperimentRole'
            },
            MinCount=nworker,
            MaxCount=nworker,
            CpuOptions={'CoreCount': 48, 'ThreadsPerCore': 1})
    assert len(instances) == nworker
    print('Waiting until instance is running...')
    for i in range(nworker):
        instances[i].wait_until_running()
        instances[i].load()
    print('Successfully launched. Connecting SSH...')

    asyncio.run(main_loop(instances, job_list, max_tries=5))

    print('Finished all jobs. Terminating worker instances...')
    for x in instances:
        x.terminate()

if __name__ == '__main__':
    main()
