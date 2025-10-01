from typing import Any, List
import os
import glob

import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from torch.utils.data import DataLoader
from llm2vec import LLM2Vec

from .base_automodel_wrapper import BaseModelSpecifications, BaseLayerwiseAutoModelWrapper
from ..misc.optimal_batch_size import find_optimal_batch_size

model_types = ["cerebras",
                "Pythia", 
                "mamba", 
                "mamba2", 
                "Medical-Llama3", 
                "Llama3", 
                "bert", 
                "roberta",
                "LLM2Vec-mntp-unsup-simcse", 
                "LLM2Vec-mntp-supervised",
                "LLM2Vec-mntp",
                "llama-instruct"]

cerebras_sizes = ['111M', '256M', '590M', '1.3B', '2.7B', '6.7B', '13B'] # '13b' also exists but doesnt fit in 24G for bfloat16
Pythia_sizes = ['14m', '70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b'] # '12b' also exists but doesnt fit in 24G for bfloat16
mamba_sizes = ['130m', '370m', '790m', '1.4b', '2.8b']
mamba2_sizes = ['130m', '370m', '780m', '1.3b', '2.7b']
bert_sizes = ['base', 'large']
medical_llama3_sizes = ['8B'] # its only 8B model
llama3_sizes = ['8B'] 
LLM2Vec_sizes = ['8B']
llama_instruct_sizes = ['8B']

model_name_to_sizes = {
    'Pythia': Pythia_sizes,
    'cerebras': cerebras_sizes,
    'mamba': mamba_sizes,
    'mamba2': mamba2_sizes,
    'Medical-Llama3': medical_llama3_sizes,
    'Llama3': llama3_sizes,
    'bert': bert_sizes,
    'roberta': bert_sizes,
    'LLM2Vec-mntp-unsup-simcse': LLM2Vec_sizes,
    'llama-instruct': llama_instruct_sizes,
    'LLM2Vec-mntp-supervised': LLM2Vec_sizes,
    'LLM2Vec-mntp': LLM2Vec_sizes,
}


def text_collate(batch):
    """
    Keep text batching independent from the heavy dataset loader module so model
    encoding does not require optional dataset-only dependencies like pyarrow.
    """
    ips = [item["input_ids"] for item in batch]
    attn = [item["attention_mask"] for item in batch]

    max_length = max(len(ip) for ip in ips)
    ips = [torch.nn.functional.pad(ip, (0, max_length - len(ip))) for ip in ips]
    attn = [torch.nn.functional.pad(mask, (0, max_length - len(mask))) for mask in attn]

    return {
        "input_ids": torch.stack(ips),
        "attention_mask": torch.stack(attn),
    }


def get_model_path(name, size):
    assert name in model_types
    if name == "cerebras":
        assert size in cerebras_sizes
        return f"cerebras/Cerebras-GPT-{size}"
    elif name == "Pythia":
        assert size in Pythia_sizes
        return f"EleutherAI/pythia-{size}"
    elif name == "Medical-Llama3":
        assert size in medical_llama3_sizes
        return f"ruslanmv/Medical-Llama3-8B"
    elif name == "Llama3":
        assert size in llama3_sizes
        return f"meta-llama/Meta-Llama-3-8B"
    elif name == "mamba":
        assert size in mamba_sizes
        return f"state-spaces/mamba-{size}-hf"
    elif name == "mamba2":
        assert size in mamba2_sizes
        return f"state-spaces/mamba2-{size}-hf" 
    elif name == "bert":
        assert size in bert_sizes
        return f"bert-{size}-uncased"
    elif name == 'roberta':
        assert size in bert_sizes
        return f"FacebookAI/roberta-{size}"
    elif name == 'LLM2Vec-mntp-unsup-simcse':
        assert size in LLM2Vec_sizes
        return f"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    elif name == 'LLM2Vec-mntp-supervised':
        assert size in LLM2Vec_sizes
        return f"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    elif name == 'LLM2Vec-mntp':
        assert size in LLM2Vec_sizes
        return f"McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    elif name == "llama-instruct":
        assert size in llama_instruct_sizes
        return f"meta-llama/Meta-Llama-3-8B-Instruct"
    else:
        raise ValueError(f"Model type {name} not found")




class TextModelSpecifications(BaseModelSpecifications):
    def __init__(self, model_family, model_size, revision, ignore_checks=False):
        super().__init__(model_family, model_size, revision, ignore_checks)
        self.model_path_func = get_model_path

    def additional_checks(self):
        if self.revision != "main":
            # Non-main revisions are supported for Pythia checkpoints.
            assert self.model_family == "Pythia"
        
        assert self.model_family in model_name_to_sizes.keys(), \
            f"Model family {self.model_family} not found, available families: {model_name_to_sizes.keys()}"
        assert self.model_size in model_name_to_sizes[self.model_family], \
            f"Model size {self.model_size} not found for model family {self.model_family}, available sizes: {model_name_to_sizes[self.model_family]}"

class TextLayerwiseAutoModelWrapper(BaseLayerwiseAutoModelWrapper):
    def __init__(self, 
                 model_specs: TextModelSpecifications, 
                 device_map="auto", 
                 evaluation_layer_idx: int = -1):
        super().__init__(model_specs, device_map, evaluation_layer_idx)

    """
    FUNCTIONS FOR INITIALIZATION
    """
    def _get_cache_dir(self):
        hf_hub_cache = os.getenv("HF_HUB_CACHE")
        if hf_hub_cache:
            return hf_hub_cache

        hf_home = os.getenv("HF_HOME")
        if hf_home:
            return os.path.join(hf_home, "hub")

        # On offline HPC hosts we often stage a shared Hugging Face cache under
        # scratch storage instead of exporting HF_HOME in every shell.
        scratch_roots = []
        scratch_env = os.getenv("SCRATCH")
        if scratch_env:
            scratch_roots.append(scratch_env)
        user = os.getenv("USER")
        if user:
            scratch_roots.append(os.path.join("/scratch", user))

        for scratch_root in scratch_roots:
            candidate = os.path.join(scratch_root, "hf_cache", "hub")
            if os.path.isdir(candidate):
                return candidate

        return None

    def _is_offline_mode(self):
        offline_flags = ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
        return any(os.getenv(flag, "").lower() in {"1", "true", "yes", "on"} for flag in offline_flags)

    def _resolve_model_path_for_loading(self):
        """
        Prefer a local snapshot when one is already cached. This avoids
        unnecessary Hub metadata requests for repo IDs and keeps loading stable
        on offline hosts.
        """
        cache_dir = self._get_cache_dir()
        if cache_dir is None:
            return self.model_path

        # If already a local path, use it as-is.
        if os.path.isdir(self.model_path):
            return self.model_path

        repo_cache_dir = os.path.join(cache_dir, f"models--{self.model_path.replace('/', '--')}")
        if not os.path.isdir(repo_cache_dir):
            return self.model_path

        revision = self.model_specs.revision
        ref_path = os.path.join(repo_cache_dir, "refs", revision)
        if os.path.isfile(ref_path):
            with open(ref_path, "r", encoding="utf-8") as f:
                commit_hash = f.read().strip()
            snapshot_dir = os.path.join(repo_cache_dir, "snapshots", commit_hash)
            if os.path.isdir(snapshot_dir):
                return snapshot_dir

        # If revision itself is a commit hash snapshot name.
        direct_snapshot = os.path.join(repo_cache_dir, "snapshots", revision)
        if os.path.isdir(direct_snapshot):
            return direct_snapshot

        # Fallback: best-effort choose a valid snapshot.
        candidates = sorted(glob.glob(os.path.join(repo_cache_dir, "snapshots", "*")))
        for candidate in reversed(candidates):
            has_config = os.path.isfile(os.path.join(candidate, "config.json"))
            has_weights = (
                os.path.isfile(os.path.join(candidate, "model.safetensors"))
                or os.path.isfile(os.path.join(candidate, "pytorch_model.bin"))
            )
            if has_config and has_weights:
                return candidate

        return self.model_path

    def _snapshot_has_tokenizer_files(self, snapshot_dir: str):
        if not os.path.isdir(snapshot_dir):
            return False

        tokenizer_candidates = (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
        )
        return any(os.path.isfile(os.path.join(snapshot_dir, name)) for name in tokenizer_candidates)

    def _resolve_tokenizer_path_for_loading(self):
        model_path_for_loading = self._resolve_model_path_for_loading()
        if self._snapshot_has_tokenizer_files(model_path_for_loading):
            return model_path_for_loading

        # Pythia checkpoints reuse the base tokenizer across revisions. Some local
        # step snapshots only cache weights/config, so fall back to the main snapshot
        # for tokenizer assets while still loading checkpoint-specific weights.
        if self.model_specs.model_family == "Pythia" and self.model_specs.revision != "main":
            cache_dir = self._get_cache_dir()
            if cache_dir is None:
                return model_path_for_loading

            repo_cache_dir = os.path.join(cache_dir, f"models--{self.model_path.replace('/', '--')}")
            if not os.path.isdir(repo_cache_dir):
                return model_path_for_loading

            main_ref_path = os.path.join(repo_cache_dir, "refs", "main")
            if os.path.isfile(main_ref_path):
                with open(main_ref_path, "r", encoding="utf-8") as f:
                    main_commit_hash = f.read().strip()
                main_snapshot_dir = os.path.join(repo_cache_dir, "snapshots", main_commit_hash)
                if self._snapshot_has_tokenizer_files(main_snapshot_dir):
                    return main_snapshot_dir

        return model_path_for_loading

    def _resolve_torch_dtype(self):
        """
        Allow runtime override of model loading dtype for startup/perf debugging.
        Supported values: auto, float16, bfloat16, float32.
        """
        override = os.getenv("IFLOW_TORCH_DTYPE", "").strip().lower()
        if override == "auto":
            return "auto"
        if override in {"float16", "fp16"}:
            return torch.float16
        if override in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if override in {"float32", "fp32"}:
            return torch.float32

        # Existing default behavior, but fall back cleanly on CPU-only hosts.
        if not torch.cuda.is_available():
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def _resolve_device_map_for_loading(self):
        if self.device_map != "auto":
            return self.device_map

        try:
            import psutil  # noqa: F401
        except ModuleNotFoundError:
            if torch.cuda.is_available():
                print("psutil not installed; falling back from device_map='auto' to {'': 0}.")
                return {"": 0}
            print("psutil not installed; falling back from device_map='auto' to CPU placement.")
            return None

        return self.device_map

    def _has_local_safetensors(self, model_path_for_loading: str):
        if not os.path.isdir(model_path_for_loading):
            return False
        return os.path.isfile(os.path.join(model_path_for_loading, "model.safetensors"))

    def setup_input_processor(self):
        tokenizer_kwargs = {}
        cache_dir = self._get_cache_dir()
        model_path_for_loading = self._resolve_tokenizer_path_for_loading()
        if cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = cache_dir
        if self._is_offline_mode():
            tokenizer_kwargs["local_files_only"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_path_for_loading, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        assert self.tokenizer.pad_token is not None

        # number of tokens the model can handle
        self.max_tokens = self.tokenizer.model_max_length

    def setup_model(self):
        cache_dir = self._get_cache_dir()
        model_path_for_loading = self._resolve_model_path_for_loading()
        config_kwargs = {
            "revision": self.model_specs.revision,
            "output_hidden_states": True,
        }
        if cache_dir is not None:
            config_kwargs["cache_dir"] = cache_dir
        if self._is_offline_mode():
            config_kwargs["local_files_only"] = True

        self.config = AutoConfig.from_pretrained(model_path_for_loading, 
                                            **config_kwargs)
        self.num_layers = self.config.num_hidden_layers + 1 
        self.update_evaluation_layer()
        self.config.num_hidden_layers = self.evaluation_layer_idx # prevents loading all layers

        FROM_PRETRAINED_KWARGS = {
            'revision': self.model_specs.revision,
            'config': self.config,
            'torch_dtype': self._resolve_torch_dtype(),
            'low_cpu_mem_usage': True,
        }
        resolved_device_map = self._resolve_device_map_for_loading()
        if resolved_device_map is not None:
            FROM_PRETRAINED_KWARGS['device_map'] = resolved_device_map
        if cache_dir is not None:
            FROM_PRETRAINED_KWARGS["cache_dir"] = cache_dir
        if self._is_offline_mode():
            FROM_PRETRAINED_KWARGS["local_files_only"] = True
        if self._has_local_safetensors(model_path_for_loading):
            FROM_PRETRAINED_KWARGS["use_safetensors"] = True

        if 'llm2vec' in self.model_path.lower():
            MODEL_CLASS = LLM2Vec
            if 'unsup' in self.model_specs.model_family.lower():
                FROM_PRETRAINED_KWARGS['peft_model_name_or_path'] = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
            elif 'supervised' in self.model_specs.model_family.lower():
                FROM_PRETRAINED_KWARGS['peft_model_name_or_path'] = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
            elif self.model_specs.model_family.lower() == 'llm2vec-mntp':
                FROM_PRETRAINED_KWARGS['peft_model_name_or_path'] = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
            else:
                raise ValueError(f"Model family {self.model_specs.model_family} not found")
        else:
            MODEL_CLASS = AutoModelForCausalLM

        self.model = MODEL_CLASS.from_pretrained(model_path_for_loading, **FROM_PRETRAINED_KWARGS).eval()        

    """
    FUNCTIONS FOR INFERENCE
    """
    @torch.no_grad()
    def encode(
        self,
        input_data: List[str],
        return_raw_hidden_states: bool = False,
        return_layerwise_encodings: bool = False,
        **kwargs: dict
    ) -> np.ndarray:
        max_sample_length = kwargs.pop("max_sample_length", 2048)
        if self.model_specs.model_family in ["bert", "roberta"]:
            max_sample_length = 512
            
        verbose = kwargs.pop("verbose", True)
        requested_batch_size = kwargs.pop("batch_size", None)
        num_workers = kwargs.pop("num_workers", 8)

        tokenized_sentences =  self.tokenizer(input_data,
                                            return_tensors="pt",
                                            padding=True,
                                            truncation=True,
                                            max_length=max_sample_length)
        
        if requested_batch_size is not None:
            # Respect caller-provided batch size (MTEB evaluators set this).
            optimal_batch_size = max(1, min(int(requested_batch_size), len(input_data)))
        else:
            # Auto-tune batch size when none is explicitly provided.
            initial_batch_size = min(512, len(input_data))
            optimal_batch_size = find_optimal_batch_size(
                model=self._get_model_with_forward_pass(),
                number_of_samples=len(input_data),
                device=self.device,
                batch_size=initial_batch_size,
                max_sentence_length=tokenized_sentences.input_ids.shape[1],
                verbose=verbose,
            )
        self.batch_size_hint = optimal_batch_size

        # create dataloader
        dataset = [{"input_ids": ids, "attention_mask": mask} 
            for ids, mask in zip(tokenized_sentences["input_ids"], 
                                tokenized_sentences["attention_mask"])]
        dataloader = DataLoader(dataset, 
                                batch_size=optimal_batch_size, 
                                shuffle=False, 
                                num_workers=num_workers, 
                                collate_fn=text_collate)

        if return_raw_hidden_states:
            embeddings, raw_hidden_states, layerwise_encodings = self._encode_helper(
                                                            dataloader,
                                                            verbose=verbose,
                                                            return_raw_hidden_states=True,
                                                            return_layerwise_encodings=True,
                                                            **kwargs)
            return np.array(embeddings), raw_hidden_states, layerwise_encodings

        if return_layerwise_encodings:
            embeddings, layerwise_encodings = self._encode_helper(
                                            dataloader,
                                            verbose=verbose,
                                            return_raw_hidden_states=False,
                                            return_layerwise_encodings=True,
                                            **kwargs)
            return np.array(embeddings), layerwise_encodings

        embeddings = self._encode_helper(dataloader,
                                        verbose=verbose,
                                        return_raw_hidden_states=False,
                                        return_layerwise_encodings=False,
                                        **kwargs) # shape: (num_samples, embedding_dim)
        return np.array(embeddings)
    
    
    def _get_model_with_forward_pass(self):
        if 'llm2vec' in self.model_path.lower():
            return self.model.model
        else:
            return self.model
    
    @torch.no_grad()
    def _encode_helper(
        self,
        dataloader,
        verbose=False,
        return_raw_hidden_states=False,
        return_layerwise_encodings=False,
        **kwargs,
    ) -> np.ndarray:
        pooling_method = kwargs.pop("pooling_method", "mean")
        encoded_batches = []
        layerwise_encoded_batches = []

        if return_raw_hidden_states:
            # can be memory intensive, so only do if needed
            raw_sample_hidden_states = []

        for batch in tqdm.tqdm(dataloader, total=len(dataloader), disable= not verbose):
            batch = self.prepare_inputs(batch)
            
            outputs = self.forward(**batch)

            hidden_states = outputs.hidden_states[self.evaluation_layer_idx]
            
            hidden_states = self._get_pooled_hidden_states(hidden_states, batch["attention_mask"], method=pooling_method)
            encoded_batches.append(hidden_states.float().cpu())
            if return_raw_hidden_states or return_layerwise_encodings:
                # Collect pooled encodings for every loaded layer without forcing
                # the caller to also keep full token-level hidden states.
                current_batch_layerwise_encodings = []
                for layer_idx in range(len(outputs.hidden_states)):
                    layer_states = outputs.hidden_states[layer_idx]
                    layer_states = self._get_pooled_hidden_states(layer_states, batch["attention_mask"], method=pooling_method)
                    current_batch_layerwise_encodings.append(layer_states.float().cpu())
                layerwise_encoded_batches.append(torch.stack(current_batch_layerwise_encodings))

            if return_raw_hidden_states:
                # Keep the expensive token-level collection on the explicit raw-hidden-state path only.
                for sample_idx in range(len(outputs.hidden_states[0])):
                    pad_idx = batch['attention_mask'][sample_idx] == 0

                    sample_hidden_states = [
                        layer_states[sample_idx][~pad_idx]
                        for layer_states in outputs.hidden_states
                    ]
                    sample_hidden_states = torch.stack(sample_hidden_states)
                    raw_sample_hidden_states.append(sample_hidden_states.squeeze().float().cpu().numpy())

        encodings = torch.cat(encoded_batches).squeeze().numpy() # shape: (num_samples, embedding_dim)
        if len(encodings.shape) == 1:
            encodings = encodings.unsqueeze(0)

        if return_raw_hidden_states:
            layerwise_encodings = torch.cat(layerwise_encoded_batches, dim=1).squeeze().numpy() # shape: (num_layers, num_samples, embedding_dim)
            return encodings, raw_sample_hidden_states, layerwise_encodings

        if return_layerwise_encodings:
            layerwise_encodings = torch.cat(layerwise_encoded_batches, dim=1).squeeze().numpy() # shape: (num_layers, num_samples, embedding_dim)
            return encodings, layerwise_encodings

        return encodings
    
    @torch.no_grad()
    def _get_pooled_hidden_states(self, hidden_states, attention_mask=None, method="mean"):
        if attention_mask is None:
            attention_mask = torch.ones_like(hidden_states[0])

        if method == "mean":
            seq_lengths = attention_mask.sum(dim=-1)
            return torch.stack(
                [
                    hidden_states[i, -length:, :].mean(dim=0)
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
            )
        elif method == "mean_including_padding":
            layer_means = torch.stack([torch.mean(x, dim=0) for x in hidden_states])
            return layer_means
        
        elif method == "last_hidden_state":
            return hidden_states[:, -1]
        elif method == "first_hidden_state":
            return hidden_states[:, 0]
        else:
            raise ValueError(f"Invalid pooling method: {method}")
        
    def prepare_inputs(self, batch):
        # move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # squeeze if needed
        if len(batch['input_ids'].shape) == 3:
            batch = {k: v.squeeze() for k, v in batch.items()}

        # unsqueeze if needed, such as for augmentation dataloaders
        if len(batch['input_ids'].shape) == 1:
            batch = {k: v.unsqueeze(0) for k, v in batch.items()}

        return batch
