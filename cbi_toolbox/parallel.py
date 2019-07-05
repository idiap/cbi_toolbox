def start_stop_indexes(job_id, n_jobs, max_index):
    """
    Computes the start and stop indexes to split array computations across jobs.

    :param job_id:
    :param n_jobs:
    :param max_index:
    :return:
    """

    if job_id > n_jobs:
        raise ValueError('Job ID larger than number of jobs!')

    if job_id < 1:
        raise ValueError('Minimum job ID is 1!')

    job_id -= 1

    if n_jobs > max_index:
        n_jobs = max_index

    min_bin_size = max_index // n_jobs
    small_bin_number = n_jobs - (max_index - min_bin_size * n_jobs)

    if job_id > n_jobs:
        raise ValueError('Amount of jobs larger than max indexes!')

    start_index = job_id * min_bin_size
    stop_index = start_index + min_bin_size

    if job_id >= small_bin_number:
        big_bins = job_id - small_bin_number
        start_index += big_bins
        stop_index += 1 + big_bins

    return start_index, stop_index
