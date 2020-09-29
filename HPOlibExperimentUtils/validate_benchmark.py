import argparse
import logging
from multiprocessing import Value
from pathlib import Path
from typing import Union, Dict

from HPOlibExperimentUtils.utils.validation_utils import write_validated_trajectory, get_unvalidated_configurations, \
    load_trajectories, load_validated_configurations

from hpolib.util.example_utils import set_env_variables_to_use_only_one_core

from HPOlibExperimentUtils.core.bookkeeper import Bookkeeper
from HPOlibExperimentUtils.utils.runner_utils import transform_unknown_params_to_dict, get_benchmark_settings, \
    load_benchmark, get_benchmark_names

logger = logging.getLogger('BenchmarkValidation')
logger.setLevel(level=logging.DEBUG)

set_env_variables_to_use_only_one_core()


def validate_benchmark(benchmark: str,
                       output_dir: Union[Path, str],
                       rng: int,
                       recompute_all: bool = False,
                       use_local: Union[bool, None] = False,
                       **benchmark_params: Dict):
    """
    After running a optimization run, often we want to validate the found configurations. To be more precisely, we want
    to validate (run on the highest budget) the configurations, which were incumbents for some time during the
    optimization process.

    The benchmarks are by default stored in singularity container which are downloaded at the first run.

    The validation script reads in the trajectory files in the output directory. Note that the validation script reads
    in all the trajectory files found in the given output path.
    Then all configurations are validated and written via the bookkeeper into a runhistory file.

    To speed up the procedure, we do not clear the validation runhistory, but keep it to check for a later validation
    run if we have already validated a specific configuration.

    Note:
    -----
    If there are more than one trajectory files found in the `output_dir`, the first one is selected for validation.
    This will change in the future. Then all found trajectories are combined and validated.

    Parameters
    ----------
    benchmark : str
        This benchmark is selected from the HPOlib3 and executed. with the optimizer from above.
        Please have a look at the experiment_settings.json file to see what benchmarks are available.
        Incomplete list of supported benchmarks:
        'cartpolereduced', 'cartpolefull',
        'xgboost',
        'learna', 'metalearna',
        'NASCifar10ABenchmark', 'NASCifar10BBenchmark', 'NASCifar10CBenchmark',
        'SliceLocalizationBenchmark', 'ProteinStructureBenchmark',
            'NavalPropulsionBenchmark', 'ParkinsonsTelemonitoringBenchmark'
    output_dir : str, Path
        Directory where the optimizer stored its results. Here should also be the trajectory file,
        which was generated in the previous step.
    rng : int, None
        Random seed for the experiment. Also changes the output directory. By default 0.
    recompute_all : bool, None
        If you want to recompute all validation results regardless of whether already precomputed results exist or not.
    use_local : bool, None
        If you want to use the HPOlib3 benchamrks in a non-containerizd version (installed locally inside the
        current python environment), you can set this parameter to True. This is not recommend.
    benchmark_params : Dict
        Some benchmarks take special parameters for the initialization. For example, The XGBOostBenchmark takes as
        input a task_id. This task_id specifies the OpenML dataset to use.

        Please take a look into the HPOlib3 Benchamarks to find out if the benchmark needs further parameter.
        Note: Most of them dont need further parameter.
    """
    logger.info(f'Start validating procedure on benchmark {benchmark}')

    output_dir = Path(output_dir)

    assert output_dir.is_dir(), f'Result folder doesn\"t exist: {output_dir}'

    unvalidated_trajectories_paths = list(output_dir.rglob(f'hpolib_trajectory.txt'))
    unvalidated_trajectories = load_trajectories(unvalidated_trajectories_paths)
    unvalidated_configurations = get_unvalidated_configurations(unvalidated_trajectories)
    validation_results = {str(configuration): -1234 for configuration in unvalidated_configurations}

    already_evaluated_configs = load_validated_configurations(output_dir)
    logger.info(f'Found {len(unvalidated_trajectories_paths)} trajectories with a total of '
                f'{len(unvalidated_configurations)} configurations (Unique: {len(validation_results)}) to validate.\n'
                f'Also, we found {len(already_evaluated_configs)} already validated configurations.')

    if not recompute_all:
        validation_results.update(already_evaluated_configs)
    logger.info('Finished loading unvalidated and already validated configurations')

    # Load and instantiate the benchmark
    benchmark_settings = get_benchmark_settings(benchmark)
    benchmark_obj = load_benchmark(benchmark_name=benchmark_settings['import_benchmark'],
                                   import_from=benchmark_settings['import_from'],
                                   use_local=use_local)

    if not use_local:
        from hpolib import config_file
        benchmark_params['container_source'] = config_file.container_source
    benchmark = benchmark_obj(rng=rng, **benchmark_params)

    # There is no reason why we want to have a wallclock time limit here.
    # Only a cutoff time limit seems to be useful.
    total_time_proxy = Value('f', 0)
    benchmark = Bookkeeper(benchmark,
                           output_dir,
                           total_time_proxy,
                           wall_clock_limit_in_s=None,
                           cutoff_limit_in_s=benchmark_settings['cutoff_in_s'],
                           is_surrogate=benchmark_settings['is_surrogate'],
                           validate=True)

    logger.debug(f'Benchmark initialized. Additional benchmark parameters {benchmark_params}')

    logger.info(f'Going to validate {len(unvalidated_configurations)} configuration')
    for i_config, configuration in enumerate(unvalidated_configurations):
        logger.info(f'[{i_config + 1:4d}|{len(unvalidated_configurations):4d}] Evaluate configuration')
        config_str = str(configuration)

        # Configuration was already validated
        if validation_results[config_str] != -1234:
            continue

        # The bookkeeper writes the trajectory automatically in to a file. But this file then contains only
        # a single entry per configuration. And the configurations are also not in the same order as the "original"
        # trajectory.
        result_dict = benchmark.objective_function_test(configuration, rng=rng)
        validation_results[config_str] = result_dict['function_value']

    benchmark.__del__()

    for unvalidated_traj, unvalidated_traj_path in zip(unvalidated_trajectories, unvalidated_trajectories_paths):
        write_validated_trajectory(unvalidated_traj, validation_results, unvalidated_traj_path, unvalidated_traj,
                                   unvalidated_traj_path)

    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='HPOlib3 Wrapper',
                                     description='HPOlib3 validated a trajectory from a benchmark with a '
                                                 'unified interface',
                                     )

    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--benchmark', choices=get_benchmark_names(), required=True, type=str)
    parser.add_argument('--rng', required=False, default=0, type=int)
    parser.add_argument('--recompute_all', action='store_true', default=False)

    args, unknown = parser.parse_known_args()
    benchmark_params = transform_unknown_params_to_dict(unknown)

    validate_benchmark(**vars(args), **benchmark_params)
