import argparse
import os
from itertools import product
from pathlib import Path
import mteb

from experiments.utils.model_definitions.text_automodel_wrapper import TextModelSpecifications, TextLayerwiseAutoModelWrapper
from experiments.utils.metrics.metric_calling import EvaluationMetricSpecifications, calculate_and_save_layerwise_metrics
from experiments.utils.misc.results_saving import construct_file_path
from experiments.utils.dataloaders.text_dataloader import get_dataloader, get_augmentation_collated_dataloader

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers.utils import logging
logging.set_verbosity_error()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_family', type=str, default='mamba')
    parser.add_argument('--model_size', type=str, default='370m')
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--evaluation_layer', type=int, default=-1, help='Layer to use for evaluation. -1 for the final layer. This is 0-indexed.')
    parser.add_argument('--base_results_path', type=str, default='experiments/results')
    parser.add_argument('--purpose', type=str, default='run_entropy_metrics', choices=['run_tasks', 'run_entropy_metrics', 'run_wikitext_metrics', 'download_datasets'])
    parser.add_argument('--raise_error', action='store_true', help='Raise immediately when a task evaluation fails.')
    parser.add_argument('--overwrite_results', action='store_true', help='Overwrite existing task result files if they already exist.')
    parser.add_argument('--metric_layer_start', type=int, default=None, help='Optional start layer (inclusive) for entropy-style metric computation.')
    parser.add_argument('--metric_layer_end', type=int, default=None, help='Optional end layer (inclusive) for entropy-style metric computation.')
    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        help=(
            "Optional comma-separated metric list for entropy runs. "
            "Supported: prompt-entropy,dataset-entropy,infonce,dime,lidar,curvature"
        ),
    )
    return parser.parse_args()


def run_entropy_metrics(
        model: TextLayerwiseAutoModelWrapper,
        model_specs: TextModelSpecifications, 
        MTEB_evaluator: mteb.MTEB,
        args
):
    default_metrics = ['prompt-entropy', 'dataset-entropy', 'infonce', 'dime', 'lidar', 'curvature']
    if args.metrics is None:
        metrics = default_metrics
    else:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        if not metrics:
            raise ValueError("--metrics was provided but no valid metric names were parsed")
        unsupported_metrics = sorted(set(metrics) - set(default_metrics))
        if unsupported_metrics:
            raise ValueError(f"Unsupported metrics requested: {unsupported_metrics}. Supported: {default_metrics}")
    print(f"Selected metrics: {metrics}")
    #metrics = ['infonce']

    if args.purpose == 'run_wikitext_metrics':
        task_datasets = ['wikitext']
        splits = ['train']
    else:
        task_datasets = [task.metadata.dataset['path'] for task in MTEB_evaluator.tasks]
        task_datasets += ['wikitext']
        splits = ['test']
        
    if model_specs.model_family in ["bert", "roberta"]:
        max_sample_length = 512
    else:
        max_sample_length = 2048

    # get maximum batch size for the model
    #optimal_batch_size = max(1, find_optimal_batch_size(model, 10000, device=model.device, max_sentence_length=max_sample_length)//2)
    #print(f"Optimal batch size: {optimal_batch_size}")

    for task_dataset, metric, split in product(task_datasets, metrics, splits):
        try:
            print(f"Running evaluation for {task_dataset} - {metric} - {split}")
            evaluation_metric_specs = EvaluationMetricSpecifications(evaluation_metric=metric)

            dataloader_kwargs = {
                'dataset_name': task_dataset,
                'split': split,
                'num_samples': 1000,
                'batch_size': 2,
                'max_sample_length': max_sample_length
            }

            # Check if results already exist, skip if they do
            results_path = construct_file_path(
                model_specs, 
                evaluation_metric_specs, 
                dataloader_kwargs, 
                args.base_results_path, 
                include_split=True
            )
            if os.path.exists(results_path):
                print(f"Results already exist for {task_dataset} - {metric} - {split}. Skipping...")
                #continue

            # Get the dataloader. Depending on the metric, might need augmentations
            if metric in ['prompt-entropy', 'dataset-entropy', 'curvature']:
                dataloader = get_dataloader(model.tokenizer, **dataloader_kwargs)
            elif metric in ['dime', 'infonce']:
                dataloader_kwargs['num_augmentations_per_sample'] = 2
                dataloader = get_augmentation_collated_dataloader(model.tokenizer, **dataloader_kwargs)
            elif metric == 'lidar':
                dataloader_kwargs['num_augmentations_per_sample'] = 16
                dataloader = get_augmentation_collated_dataloader(model.tokenizer, **dataloader_kwargs)
            else:
                raise ValueError(f"dataloader for metric {metric} is not implemented yet")

            # compute the metrics for the dataloader
            calculate_and_save_layerwise_metrics(
                model,
                dataloader,
                model_specs,
                evaluation_metric_specs,
                dataloader_kwargs,
                metric_layer_start=args.metric_layer_start,
                metric_layer_end=args.metric_layer_end,
            )
    
        except Exception as e:
            if 'SplitDoesNotExist' in str(e):
                print(f"The dataset {task_dataset} does not have split {split}. Skipping {metric} computation for this dataset/split...")
                continue
            if (
                task_dataset == 'wikitext'
                and (
                    "Couldn't reach the Hugging Face Hub for dataset 'wikitext'" in str(e)
                    or "offline mode is enabled" in str(e)
                )
            ):
                print(
                    "Wikitext is unavailable in offline mode for this environment. "
                    "Skipping wikitext metric computation."
                )
                continue

            print(f"Error running evaluation for {task_dataset} - {metric} - {split}: {str(e)}")
            if args.raise_error:
                raise e

def main():
    args = parse_args()
    model_family = args.model_family
    model_size = args.model_size
    revision = args.revision
    evaluation_layer = args.evaluation_layer

    print(f"Running evaluation for {model_family} {model_size} {revision} layer {evaluation_layer}")
    model_specs = TextModelSpecifications(model_family, model_size, revision=revision)

    # handle tasks
    mteb_eng = mteb.get_benchmark("MTEB(eng)")
    reduced_mteb_eng_tasks = [task for task in mteb_eng if task.metadata.category != 'p2p']
    reduced_mteb_eng_tasks = [task for task in reduced_mteb_eng_tasks if task.metadata.type != 'Retrieval']
    evaluator = mteb.MTEB(tasks=reduced_mteb_eng_tasks)
    
    device_map = "auto" if model_family != 'bert' else None
    model = TextLayerwiseAutoModelWrapper(model_specs, device_map=device_map, evaluation_layer_idx=evaluation_layer)

    # if BERT, we need to manually move the model to the device because the device map is not supported
    # https://github.com/huggingface/transformers/issues/25296
    if model_family == 'bert':
        model.model = model.model.to("cuda:0")

    if args.purpose == 'run_tasks': 
        results_output_folder = f'{args.base_results_path}/{model_family}/{model_size}/{revision}/mteb/layer_{model.evaluation_layer_idx}'
        def custom_create_output_folder(*args):
            output_folder = Path(results_output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
            return output_folder
        
        encoding_kwargs = {'verbose': True, 'batch_size': 32}
        evaluator.create_output_folder = custom_create_output_folder
        evaluator.run(model, 
                      encode_kwargs=encoding_kwargs, 
                      output_folder='./mteb-results', 
                      raise_error=args.raise_error,
                      overwrite_results=args.overwrite_results, 
                      verbosity=2)

    elif args.purpose == 'run_entropy_metrics' or args.purpose == 'run_wikitext_metrics':
        run_entropy_metrics(model, model_specs, evaluator, args)

    elif args.purpose == 'download_datasets':
        for task in evaluator.tasks:
            task.load_data()


if __name__ == "__main__":
    main()
