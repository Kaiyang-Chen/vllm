# ruff: noqa: SIM117
import copy
import glob
import time
import os
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from typing import (TYPE_CHECKING, Any, Dict, Generator, List, Optional, Tuple,
                    Type, ContextManager)

import torch
from torch import nn

from vllm.config import (VLLM_USE_MODELSCOPE, DeviceConfig, LoadConfig,
                         LoadFormat, LoRAConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, VisionLanguageConfig)
from vllm.logger import init_logger
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig, is_vllm_serialized_tensorizer, load_with_tensorizer,
    tensorizer_weights_iterator)
from vllm.model_executor.model_loader.utils import (get_model_architecture,
                                                    set_default_torch_dtype)
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf, filter_files_not_needed_for_inference,
    get_quant_config, initialize_dummy_weights, np_cache_weights_iterator,
    pt_weights_iterator, safetensors_weights_iterator)
from vllm.model_executor.models.llava import LlavaForConditionalGeneration

if TYPE_CHECKING:
    from vllm.model_executor.layers.linear import LinearMethodBase

_VISION_MODEL_CLASSES = [
    LlavaForConditionalGeneration,
]

logger = init_logger(__name__)


def _get_linear_method(
        model_config: ModelConfig,
        load_config: LoadConfig) -> Optional["LinearMethodBase"]:
    """Get the (maybe quantized) linear method."""
    linear_method = None
    if model_config.quantization is not None:
        quant_config = get_quant_config(model_config, load_config)
        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        if capability < quant_config.get_min_capability():
            raise ValueError(
                f"The quantization method {model_config.quantization} is not "
                "supported for the current GPU. "
                f"Minimum capability: {quant_config.get_min_capability()}. "
                f"Current capability: {capability}.")
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}")

        linear_method = quant_config.get_linear_method()
    return linear_method


def _get_model_initialization_kwargs(
        model_class: Type[nn.Module], lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig]
) -> Dict[str, Any]:
    """Get extra kwargs for model initialization."""
    extra_kwargs = {}
    if hasattr(model_class, "supported_lora_modules"):
        extra_kwargs["lora_config"] = lora_config
    elif lora_config:
        raise ValueError(
            f"Model {model_class.__name__} does not support LoRA, "
            "but LoRA is enabled. Support for this model may "
            "be added in the future. If this is important to you, "
            "please open an issue on github.")
    elif model_class in _VISION_MODEL_CLASSES:
        extra_kwargs["vision_language_config"] = vision_language_config
    return extra_kwargs


def _initialize_model(
        model_config: ModelConfig, load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig]) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_class = get_model_architecture(model_config)[0]
    linear_method = _get_linear_method(model_config, load_config)

    return model_class(config=model_config.hf_config,
                       linear_method=linear_method,
                       **_get_model_initialization_kwargs(
                           model_class, lora_config, vision_language_config))


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig) -> nn.Module:
        """Load a model with the given configurations."""
        ...


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def _maybe_download_from_modelscope(
            self, model: str, revision: Optional[str]) -> Optional[str]:
        """Download model from ModelScope hub if VLLM_USE_MODELSCOPE is True.
        
        Returns the path to the downloaded model, or None if the model is not
        downloaded from ModelScope."""
        if VLLM_USE_MODELSCOPE:
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            # pylint: disable=C.
            from modelscope.hub.snapshot_download import snapshot_download

            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=self.load_config.download_dir,
                    revision=revision)
            else:
                model_path = model
            return model_path
        return None

    def _prepare_weights(self, model_name_or_path: str,
                         revision: Optional[str],
                         fall_back_to_pt: bool) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = self._maybe_download_from_modelscope(
            model_name_or_path, revision) or model_name_or_path

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        # Some quantized models use .pt files for storing the weights.
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]
        elif load_format == LoadFormat.SAFETENSORS:
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == LoadFormat.PT:
            allow_patterns = ["*.pt"]
        elif load_format == LoadFormat.NPCACHE:
            allow_patterns = ["*.bin"]
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if not is_local:
            hf_folder = download_weights_from_hf(model_name_or_path,
                                                 self.load_config.download_dir,
                                                 allow_patterns)
        else:
            hf_folder = model_name_or_path

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if not use_safetensors:
            hf_weights_files = filter_files_not_needed_for_inference(
                hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`")

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, model_name_or_path: str, revision: Optional[str],
        fall_back_to_pt: bool
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision, fall_back_to_pt)
        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            return np_cache_weights_iterator(model_name_or_path,
                                             self.load_config.download_dir,
                                             hf_folder, hf_weights_files)
        if use_safetensors:
            return safetensors_weights_iterator(hf_weights_files)
        return pt_weights_iterator(hf_weights_files)

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig) -> nn.Module:
        start_time = time.perf_counter()
        init_contexts = [no_init_weights(_enable=True)]
        init_contexts.append(init_empty_weights())
        with set_default_torch_dtype(model_config.dtype):
            with ContextManagers(init_contexts):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config)
            model.load_weights(
                self._get_weights_iterator(model_config.model,
                                           model_config.revision,
                                           fall_back_to_pt=getattr(
                                               model,
                                               "fall_back_to_pt_during_load",
                                               True)), )
        model.to(device_config.device)
        end_time = time.perf_counter()

        # Calculate the difference
        elapsed_time = end_time - start_time
        print(f"The function took {elapsed_time} seconds to run.")
        return model.eval()
    
@contextmanager
def init_empty_weights(include_buffers: bool = None):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    Make sure to overwrite the default device_map param for [`load_checkpoint_and_dispatch`], otherwise dispatch is not
    called.

    </Tip>
    """
    if include_buffers is None:
        include_buffers = False
    with init_on_device(torch.device("meta"), include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = None):
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_on_device

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Liner(100, 100)  # on `cuda` device
    ```
    """
    # if include_buffers is None:
    #     include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)
    include_buffers = False


    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)

    
class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)



TORCH_INIT_FUNCTIONS = {
    "uniform_": nn.init.uniform_,
    "normal_": nn.init.normal_,
    "trunc_normal_": nn.init.trunc_normal_,
    "constant_": nn.init.constant_,
    "xavier_uniform_": nn.init.xavier_uniform_,
    "xavier_normal_": nn.init.xavier_normal_,
    "kaiming_uniform_": nn.init.kaiming_uniform_,
    "kaiming_normal_": nn.init.kaiming_normal_,
    "uniform": nn.init.uniform,
    "normal": nn.init.normal,
    "xavier_uniform": nn.init.xavier_uniform,
    "xavier_normal": nn.init.xavier_normal,
    "kaiming_uniform": nn.init.kaiming_uniform,
    "kaiming_normal": nn.init.kaiming_normal,
}

@contextmanager
def no_init_weights(_enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.

    """

    if _enable:

        def _skip_init(*args, **kwargs):
            pass

        # # Save the original initialization functions
        for name, init_func in TORCH_INIT_FUNCTIONS.items():
            setattr(torch.nn.init, name, _skip_init)
    try:
        yield
    finally:
        if _enable:
            # # Restore the original initialization functions
            for name, init_func in TORCH_INIT_FUNCTIONS.items():
                setattr(torch.nn.init, name, init_func)

class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(f"Model loader extra config is not supported for "
                             f"load format {load_config.load_format}")

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig) -> nn.Module:
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config)
            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)
        return model.eval()


class TensorizerLoader(BaseModelLoader):
    """Model loader using CoreWeave's tensorizer library."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if isinstance(load_config.model_loader_extra_config, TensorizerConfig):
            self.tensorizer_config = load_config.model_loader_extra_config
        else:
            self.tensorizer_config = TensorizerConfig(
                **load_config.model_loader_extra_config)

    def _verify_config(self, model_config: ModelConfig,
                       parallel_config: ParallelConfig):
        self.tensorizer_config.verify_with_model_config(model_config)
        self.tensorizer_config.verify_with_parallel_config(parallel_config)

    def _get_weights_iterator(
            self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        tensorizer_args = self.tensorizer_config._construct_tensorizer_args()
        return tensorizer_weights_iterator(tensorizer_args)

    def _load_model_unserialized(
            self, model_config: ModelConfig, device_config: DeviceConfig,
            lora_config: Optional[LoRAConfig],
            vision_language_config: Optional[VisionLanguageConfig]
    ) -> nn.Module:
        """Load an unserialized model with tensorizer.

        Unserialized here means "not serialized with tensorizer". This
        should still be faster than default HuggingFace loading, but will
        be slower than loading a tensorizer-serialized model.
        """
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config,
                                          lora_config, vision_language_config)

            model.load_weights(self._get_weights_iterator())
        return model.eval()

    def _load_model_serialized(
            self, model_config: ModelConfig, device_config: DeviceConfig,
            lora_config: Optional[LoRAConfig],
            vision_language_config: Optional[VisionLanguageConfig]
    ) -> nn.Module:
        """Load a serialized model with tensorizer.

        See the examples/tensorize_vllm_model.py example "
        script for serializing vLLM models."""
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model_class = get_model_architecture(model_config)[0]
                linear_method = _get_linear_method(model_config,
                                                   self.load_config)
                extra_kwargs = _get_model_initialization_kwargs(
                    model_class, lora_config, vision_language_config)
                extra_kwargs["linear_method"] = linear_method

                tensorizer_config = copy.copy(self.tensorizer_config)
                tensorizer_config.model_class = model_class
                tensorizer_config.hf_config = model_config.hf_config
                tensorizer_config.dtype = model_config.dtype

                model = load_with_tensorizer(tensorizer_config, **extra_kwargs)
        return model.eval()

    def load_model(self, *, model_config: ModelConfig,
                   device_config: DeviceConfig,
                   lora_config: Optional[LoRAConfig],
                   vision_language_config: Optional[VisionLanguageConfig],
                   parallel_config: ParallelConfig,
                   scheduler_config: SchedulerConfig) -> nn.Module:
        self._verify_config(model_config, parallel_config)

        if is_vllm_serialized_tensorizer(self.tensorizer_config):
            return self._load_model_serialized(model_config, device_config,
                                               lora_config,
                                               vision_language_config)
        return self._load_model_unserialized(model_config, device_config,
                                             lora_config,
                                             vision_language_config)


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.DUMMY:
        return DummyModelLoader(load_config)

    if load_config.load_format == LoadFormat.TENSORIZER:
        return TensorizerLoader(load_config)

    return DefaultModelLoader(load_config)
