import gc
import torch

try:
    from ..model_definitions.vision_automodel_wrapper import VisionLayerwiseAutoModelWrapper
except ModuleNotFoundError:
    VisionLayerwiseAutoModelWrapper = ()

try:
    from ..model_definitions.probe.LinearProbe import LinearModel
except ModuleNotFoundError:
    LinearModel = ()

def garbage_collect_cuda():
    gc.collect()
    torch.cuda.empty_cache()

def is_oom_error(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )

@torch.no_grad()
def find_optimal_batch_size(model, number_of_samples, device, batch_size=512, max_sentence_length=2048, max_trials=10, verbose=False):
    """ Batch scaling mode where the size is doubled at each iteration until an
        OOM error is encountered. 

        Adapted from the Pytorch Lightning implementation of auto batch size
        https://github.com/Lightning-AI/pytorch-lightning/pull/1638/files    
    """
    original_batch_size = batch_size
    had_success = False
    for _ in range(max_trials):
        garbage_collect_cuda()
        try:
            if isinstance(model, VisionLayerwiseAutoModelWrapper) or isinstance(model, LinearModel):
                model_family = model.model_specs.model_family if isinstance(model, VisionLayerwiseAutoModelWrapper) else model.backbone.model_specs.model_family
                if model_family == 'i-jepa':
                    worst_case_batch = {
                        "x": torch.ones((batch_size, 3, 224, 224), dtype=model.dtype).to(device),
                    }
                else:
                    worst_case_batch = {
                        "pixel_values": torch.ones((batch_size, 3, 224, 224), dtype=model.dtype).to(device),
                    }
            else:
                worst_case_batch = {
                    "input_ids": torch.ones((batch_size, max_sentence_length), dtype=torch.int).to(device),
                    "attention_mask": torch.ones((batch_size, max_sentence_length), dtype=torch.int).to(device)
                }

            model(**worst_case_batch)
            had_success = True

            if batch_size > number_of_samples:
                break

            batch_size *= 2

        except RuntimeError as exception:
            if is_oom_error(exception):
                batch_size = batch_size // 2
                if had_success:
                    break
            else:
                raise  # some other error not memory related

    if verbose:
        #pass # todo: add this back when its not so annoying
        print(f"Starting batch size: {original_batch_size}, Optimal batch size: {batch_size}, max_sentence_length: {max_sentence_length}")

    garbage_collect_cuda()
    return batch_size
