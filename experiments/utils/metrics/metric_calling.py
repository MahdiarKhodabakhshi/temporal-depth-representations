from typing import Any, Callable, List, Literal, Type, Dict, Union
import torch
import tqdm
import numpy as np

import repitl.matrix_itl as itl
import repitl.difference_of_entropies as dent

from . import metric_functions as mf
from ..model_definitions.base_automodel_wrapper import BaseModelSpecifications
from ..model_definitions.text_automodel_wrapper import TextLayerwiseAutoModelWrapper
DISABLE_TQDM = False

metric_name_to_function = {
    'entropy': mf.compute_entropy,
    'lidar': mf.compute_lidar,
    'dime': mf.compute_dime,
    'infonce': mf.compute_infonce,
    'curvature': mf.compute_curvature,
    'intrinsic_dimension': mf.compute_intrinsic_dimension,
}

class EvaluationMetricSpecifications:
    def __init__(
        self, 
        evaluation_metric, 
        num_samples = 1000, 
        alpha = 1, 
        normalizations = ['maxEntropy', 'raw', 'logN', 'logNlogD', 'logD'],
        curvature_k = 1
    ):
        self.evaluation_metric = evaluation_metric
        self.num_samples = num_samples

        
        if self.evaluation_metric == 'prompt-entropy':
            self.granularity = 'prompt'
            self.evaluation_metric = 'entropy'
        elif self.evaluation_metric == 'dataset-entropy':
            self.granularity = 'dataset'
            self.evaluation_metric = 'entropy'
        else:
            self.granularity = None

        # for matrix-based metrics (LIDAR, DIME, entropy)
        self.normalizations = normalizations
        self.alpha = alpha

        # for curvature
        self.curvature_k = curvature_k
        
        self.do_checks()

    def do_checks(self):
        assert self.evaluation_metric in metric_name_to_function.keys()
        assert self.granularity in ['prompt', 'dataset', None]

        assert self.alpha > 0
        assert self.num_samples > 0
        assert self.curvature_k > 0 and isinstance(self.curvature_k, int)

    def __str__(self):
        return f"Metric: {self.evaluation_metric}"

def compute_per_forward_pass(
    model,
    dataloader,
    compute_function,
    should_average_over_layers=True,
    layer_start=None,
    layer_end=None,
    **kwargs,
):
    """
    Compute a metric for each forward pass through the model.

    Args:
        model (torch.nn.Module): The model to use for forward passes.
        dataloader (torch.utils.data.DataLoader): The dataloader providing batches.
        compute_function (callable): The function to compute the metric.
        **kwargs: Additional keyword arguments to pass to compute_function.

    Returns:
        dict: A dictionary of computed metrics, averaged over all samples.
    """
    results = {}
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, total=len(dataloader), disable=DISABLE_TQDM, desc="Processing batches"):
            batch = model.prepare_inputs(batch)
            outputs = model(**batch)
            
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
            elif isinstance(outputs, dict) and 'hidden_states' in outputs:
                hidden_states = outputs['hidden_states']
            else:
                hidden_states = outputs

            for sample_idx in range(len(hidden_states[0])):
                if 'attention_mask' in batch.keys():
                    # ignore padding tokens
                    pad_idx = batch['attention_mask'][sample_idx] == 0
                else:
                    pad_idx = None

                sample_hidden_states = [
                    mf.normalize(layer_states[sample_idx][~pad_idx]) if pad_idx is not None
                    else mf.normalize(layer_states[sample_idx])
                    for layer_states in hidden_states
                ]
                sample_hidden_states = torch.stack(sample_hidden_states) # L x NUM_TOKENS x D
                if layer_start is not None or layer_end is not None:
                    start_idx = 0 if layer_start is None else int(layer_start)
                    end_idx = sample_hidden_states.shape[0] - 1 if layer_end is None else int(layer_end)
                    sample_hidden_states = sample_hidden_states[start_idx : end_idx + 1]

                sample_result = compute_function(sample_hidden_states, **kwargs)
                for norm, values in sample_result.items():
                    if norm not in results:
                        results[norm] = []
                    results[norm].append(values)

    if should_average_over_layers:
        return {norm: np.array(values).mean(axis=0) for norm, values in results.items()}
    else:
        return {norm: np.array(values) for norm, values in results.items()}

def compute_on_concatenated_passes(
    model,
    dataloader,
    compute_function,
    layer_start=None,
    layer_end=None,
    **kwargs,
):
    """
    Compute a metric on concatenated hidden states from multiple forward passes.

    Args:
        model (torch.nn.Module): The model to use for forward passes.
        dataloader (torch.utils.data.DataLoader): The dataloader providing batches.
        compute_function (callable): The function to compute the metric.
        **kwargs: Additional keyword arguments to pass to compute_function.

    Returns:
        dict: A dictionary of computed metrics.
    """
    all_hidden_states = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, disable=DISABLE_TQDM):
            if not isinstance(batch, tuple):
                if isinstance(model, TextLayerwiseAutoModelWrapper):
                    batch = (batch,)
                elif len(batch) == 3:
                    # for vision model case
                    batch = (batch,)
            
            batch_hidden_states = []
            for sub_batch in batch:
                sub_batch = model.prepare_inputs(sub_batch)
               
                outputs = model(**sub_batch)
                if 'attention_mask' in sub_batch.keys():
                    attention_mask = sub_batch['attention_mask']
                else:
                    attention_mask = None

                layerwise_mean_tokens = [
                    model._get_pooled_hidden_states(layer_states, attention_mask, method='mean')
                    for layer_states in outputs.hidden_states
                ]  # L x BS x D
                # Keep batch dimension for single-sample batches.
                layerwise_mean_tokens = [mf.normalize(x if x.dim() == 2 else x.unsqueeze(0)) for x in layerwise_mean_tokens] # L x BS x D
                layerwise_mean_tokens = torch.stack(layerwise_mean_tokens) # L x BS x D

                if len(layerwise_mean_tokens.shape) == 2:
                    layerwise_mean_tokens = layerwise_mean_tokens.unsqueeze(1) # L x BS x D

                batch_hidden_states.append(layerwise_mean_tokens)
            
            all_hidden_states.append(torch.stack(batch_hidden_states)) # NUM_AUG x L x BS x D
 
    concatenated_states = torch.cat(all_hidden_states, dim=2) # NUM_AUG x L x NUM_SAMPLES x D
    concatenated_states = concatenated_states.permute(1, 2, 0, 3) # L x NUM_SAMPLES x NUM_AUG x D
    if layer_start is not None or layer_end is not None:
        start_idx = 0 if layer_start is None else int(layer_start)
        end_idx = concatenated_states.shape[0] - 1 if layer_end is None else int(layer_end)
        concatenated_states = concatenated_states[start_idx : end_idx + 1]
    # For entropy, NUM_AUG is 1. Remove only that axis and keep sample dimension.
    if concatenated_states.shape[2] == 1:
        concatenated_states = concatenated_states[:, :, 0, :]
    print(concatenated_states.shape)
    return compute_function(concatenated_states, **kwargs)


def calculate_and_save_layerwise_metrics(
    model,
    dataloader,
    model_specs: BaseModelSpecifications,
    evaluation_metric_specs: EvaluationMetricSpecifications,
    dataloader_kwargs: Dict[str, Any],
    should_save_results: bool = True,
    metric_layer_start: int | None = None,
    metric_layer_end: int | None = None,
):
    num_layers = getattr(model, "num_layers", None)
    selected_start = metric_layer_start
    selected_end = metric_layer_end
    if selected_start is not None or selected_end is not None:
        if num_layers is None:
            raise ValueError("Cannot apply metric layer range without model.num_layers")
        selected_start = 0 if selected_start is None else int(selected_start)
        selected_end = (num_layers - 1) if selected_end is None else int(selected_end)
        assert 0 <= selected_start <= selected_end < num_layers, (
            f"Invalid metric layer range [{selected_start}, {selected_end}] for num_layers={num_layers}"
        )

    if evaluation_metric_specs.evaluation_metric == 'entropy':
        compute_func_kwargs = {
            'alpha': evaluation_metric_specs.alpha,
            'normalizations': evaluation_metric_specs.normalizations
        }
        forward_pass_func = compute_per_forward_pass if evaluation_metric_specs.granularity == 'prompt' else compute_on_concatenated_passes
  

    elif evaluation_metric_specs.evaluation_metric == 'curvature':
        compute_func_kwargs = {
            'k': evaluation_metric_specs.curvature_k
        }
        forward_pass_func = compute_per_forward_pass

    elif evaluation_metric_specs.evaluation_metric == 'lidar':
        compute_func_kwargs = {
            'alpha': evaluation_metric_specs.alpha,
            'normalizations': evaluation_metric_specs.normalizations,
        }
        forward_pass_func = compute_on_concatenated_passes

    elif evaluation_metric_specs.evaluation_metric == 'dime':
        compute_func_kwargs = {
            'alpha': evaluation_metric_specs.alpha,
            'normalizations': evaluation_metric_specs.normalizations,
        }
        forward_pass_func = compute_on_concatenated_passes

    elif evaluation_metric_specs.evaluation_metric == 'infonce':
        compute_func_kwargs = {
            'temperature': 0.1,
        }
        forward_pass_func = compute_on_concatenated_passes
    
    elif evaluation_metric_specs.evaluation_metric == 'intrinsic_dimension':
        compute_func_kwargs = {}
        forward_pass_func = compute_per_forward_pass

    compute_func = metric_name_to_function[evaluation_metric_specs.evaluation_metric]
    results = forward_pass_func(
        model,
        dataloader,
        compute_func,
        layer_start=selected_start,
        layer_end=selected_end,
        **compute_func_kwargs,
    )

    # Keep layer indexing aligned with full model depth by padding uncomputed layers with NaN.
    if selected_start is not None and num_layers is not None:
        padded_results = {}
        for norm, values in results.items():
            values_arr = np.asarray(values, dtype=float)
            expected = selected_end - selected_start + 1
            if values_arr.shape[0] != expected:
                raise ValueError(
                    f"Unexpected number of layer values for {norm}: got {values_arr.shape[0]}, expected {expected}"
                )
            full_values = np.full((num_layers,), np.nan, dtype=float)
            full_values[selected_start : selected_end + 1] = values_arr
            padded_results[norm] = full_values
        results = padded_results

    if should_save_results:
        from ..misc.results_saving import save_results # here to avoid circular imports
        save_results(results, model_specs, evaluation_metric_specs, dataloader_kwargs)

    return results
