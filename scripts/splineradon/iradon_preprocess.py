import argparse
import json
import numpy as np
import os

import cbi_toolbox as cbi
from cbi_toolbox import parallel

SINO_SUFFIX = 'sino'
PREPROCESS_SUFFIX = 'pre'
PARTS_PREFIX = 'parts'


def preprocess(args, out_file_name, output_dir, sino_file):
    n_jobs = args.n_jobs
    job_id = args.job_id

    if n_jobs > 1:
        output_dir = os.path.join(output_dir, '{}_{}'.format(PARTS_PREFIX, PREPROCESS_SUFFIX))
    os.makedirs(output_dir, exist_ok=True)

    if n_jobs > 1:
        out_file_name = '{}_{:04d}'.format(out_file_name, job_id)

    out_file_name = '{}.npy'.format(out_file_name)

    output_file = os.path.join(output_dir, out_file_name)

    if os.path.exists(output_file):
        print('Result exists in {}'.format(output_file))
        if args.overwrite:
            print('Overwriting')
        else:
            print('Skipping')
            exit(0)

    print('Loading file {}'.format(sino_file))
    sinogram = np.load(sino_file, mmap_mode='r')

    start_i, stop_i = parallel.start_stop_indexes(job_id, n_jobs, sinogram.shape[2])
    sinogram = sinogram[..., start_i:stop_i]

    sinogram = sinogram.astype(np.float64)

    step = args.split
    print('Running pre-filtering, step {}'.format(step))

    for index in range(sinogram.shape[2] // step):
        # TODO give spline orders as parameter (load from json)
        sinogram[..., step * index:step * index + step] = cbi.splineradon.steps.spliradon_pre(
            sinogram[..., step * index:step * index + step])

    remaining = sinogram.shape[2] % step
    if remaining != 0:
        sinogram[-remaining:, ...] = cbi.splineradon.steps.spliradon_pre(
            sinogram[-remaining, ...])

    print('Saving to {}'.format(output_file))
    np.save(output_file, sinogram)


def merge(args, out_file_name, output_dir):
    n_jobs = args.n_jobs
    if n_jobs <= 1:
        print('Nothing to merge')
        exit(0)

    part_dir = os.path.join(output_dir, '{}_{}'.format(PARTS_PREFIX, PREPROCESS_SUFFIX))
    part_file_base = os.path.join(part_dir, out_file_name)

    output_file = os.path.join(output_dir, '{}.npy'.format(out_file_name))

    arrays = []

    for part in range(n_jobs):
        part_file_name = '{}_{:04d}.npy'.format(part_file_base, part + 1)
        print('Loading file {}'.format(part_file_name))
        arrays.append(np.load(part_file_name))

    merged = np.concatenate(arrays, axis=-1)

    print('Saving to {}'.format(output_file))
    np.save(output_file, merged)

    import shutil
    print('Removing {}'.format(part_dir))
    shutil.rmtree(part_dir)


def main():
    try:
        def_job_id = os.environ['SGE_TASK_ID']
    except KeyError:
        def_job_id = 1

    parser = argparse.ArgumentParser()

    parser.add_argument('conf', metavar='CONFIG_FILE.json', help='json config file', type=str)
    parser.add_argument('-j', '--job_id', metavar='JOB_ID', type=int, default=def_job_id)
    parser.add_argument('-n', '--n_jobs', type=int, help='number of jobs launched', default=1)
    parser.add_argument('--split', metavar='STEP', type=int, help='size for sub-array splitting', default=5)
    parser.add_argument('--merge', action='store_true', help='merge arrays computed as parts')
    parser.add_argument('--overwrite', action='store_true', help='overwrite if results exist')

    args = parser.parse_args()

    print('Loading config file {}'.format(args.conf))
    with open(args.conf, 'r') as conf_file:
        conf_dict = json.load(conf_file)
    print(conf_dict)

    name = conf_dict['name']
    output_dir = os.path.join(conf_dict['output_path'], name)
    sino_file = os.path.join(output_dir, '{}_{}.npy'.format(name, SINO_SUFFIX))
    out_file_name = '{}_{}'.format(name, PREPROCESS_SUFFIX)

    out_file = os.path.join(output_dir, '{}.npy'.format(out_file_name))
    if os.path.exists(out_file):
        print('Result exists in {}'.format(out_file))
        if args.overwrite:
            print('Overwriting')
            os.remove(out_file)
        else:
            print('Skipping')
            exit(0)

    if args.merge:
        print('Merging')
        merge(args, out_file_name, output_dir)
    else:
        print('Preprocessing')
        preprocess(args, out_file_name, output_dir, sino_file)


if __name__ == '__main__':
    main()
