import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
)
import numpy as np

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class DTSMLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.AdaLora`].

    Args:
        orth_reg_weight (`float`): The coefficient of orthogonal regularization.
    """
    orth_reg_weight: float = field(default=0.0, metadata={"help": "The orthogonal regularization coefficient."})

    def __post_init__(self):
        self.peft_type = PeftType.DTSMLORA


class DTSMLoraModel(LoraModel):
    """
    Creates DTSMLoRA model from a pretrained transformers model. Paper:
    https://arxiv.org/pdf/2501.08008

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`DTSMLoraConfig`]): The configuration of the AdaLora model.

    Returns:
        `torch.nn.Module`: The AdaLora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import DTSMLoraModel, DTSMLoraConfig
        >>> config = DTSMLoraConfig(
                peft_type="DTSMLORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
                lora_dropout=0.01,
            )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> model = DTSMLoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`DTSMLoraConfig`]): The configuration of the AdaLora model.
    """

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_dtsmlora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "DTSMLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "DTSMLoraModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except the one you want to train."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)

        if (loaded_in_8bit or loaded_in_4bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, LoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        kwargs.update(
                            {
                                "has_fp16_weights": target.state.has_fp16_weights,
                                "memory_efficient_backward": target.state.memory_efficient_backward,
                                "threshold": target.state.threshold,
                                "index": target.index,
                            }
                        )
                        new_module = DTSMLinear8bitLt(
                            adapter_name, target.in_features, target.out_features, bias=bias, **kwargs
                        )
                    elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
                        fourbit_kwargs = kwargs.copy()
                        fourbit_kwargs.update(
                            {
                                "compute_dtype": target.compute_dtype,
                                "compress_statistics": target.weight.compress_statistics,
                                "quant_type": target.weight.quant_type,
                            }
                        )
                        new_module = DTSMLinear4bit(
                            adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs
                        )
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = target.in_features, target.out_features
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = DTSMLinear(adapter_name, in_features, out_features, bias=bias, **kwargs)

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)
        orth_reg_weight = self.peft_config[self.trainable_adapter_name].orth_reg_weight
        if orth_reg_weight == 0:
            return outputs
        # Calculate the orthogonal regularization
        print("=====================================Enter the orthogonal regularization=====================================")
        if hasattr(outputs, "loss"):
            regu_loss = 0
            num_param = 0
            for n, p in self.model.named_parameters():
                if ("lora_A" in n or "lora_B" in n) and self.trainable_adapter_name in n:
                    para_cov = p @ p.T if "lora_A" in n else p.T @ p
                    I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
                    I.requires_grad = False
                    num_param += 1
                    regu_loss += torch.norm(para_cov - I, p="fro")
            if num_param > 0:
                regu_loss = regu_loss / num_param
            else:
                regu_loss = 0
            outputs.loss += orth_reg_weight * regu_loss
        return outputs

    def update_and_increase(self, global_step, optimizer):
        self.rankallocator.update_and_increase(self.model, global_step, optimizer)


    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    print(f"start merging DTSMLoRA key: {key}")
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    @staticmethod
    def _prepare_dtsmlora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        return peft_config


class DTSMLoraLayer(LoraLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__(in_features, out_features)
        self.lora_C = nn.ParameterDict({})
        self.lora_D = nn.ParameterDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.ranknum = {}
        self.score = {}
        self.preNormValue = {}

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:

            def lora_dropout_layer(x):
                return x

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros(r, self.in_features))}))
            self.lora_C.update(nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros(r, r))}))
            self.lora_D.update(nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros(r, r))}))
            self.lora_B.update(nn.ParameterDict({adapter_name: nn.Parameter(self.weight.new_zeros(self.out_features, r))}))
            self.W = loraW()
            self.gradMatrix_trace = 0
            # The current rank
            self.ranknum[adapter_name] = 0
            self.score[adapter_name] = 0.0
            self.preNormValue[adapter_name] = 0.0
            self.scaling[adapter_name] = lora_alpha / r
            print("=====================================Enter the new DTSMLoraLayer linear method========================================")

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            print("==================================Enter the reset_lora_parameters method=====================================")
            nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_C[adapter_name])
            nn.init.zeros_(self.lora_D[adapter_name])
            nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.02)


class loraW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, C, D, B, scaling):
        return B @ ((C + D) @ A) * scaling

class DTSMLinear(nn.Linear, DTSMLoraLayer):
    # Triangular Adaptive Low-Rank Adaptation
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        DTSMLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name


    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r[self.active_adapter] > 0:
            print("==============start DTSMLoRA merging==============")
            self.weight.data += T(
                self.W(self.lora_A[self.active_adapter], self.lora_C[self.active_adapter],
                       self.lora_D[self.active_adapter], self.lora_B[self.active_adapter],
                       self.scaling[self.active_adapter])
            )
            self.merged = True

    def unmerge(self):
        print("start DTSMLoRA unmerging")
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r[self.active_adapter] > 0:
            self.weight.data -= T(
                self.W(self.lora_A[self.active_adapter], self.lora_C[self.active_adapter],
                       self.lora_D[self.active_adapter], self.lora_B[self.active_adapter],
                       self.scaling[self.active_adapter])
            )
            self.merged = False

    def forward(self, x: torch.Tensor):
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            result += (
                self.lora_dropout[self.active_adapter](x)
                @ self.W(self.lora_A[self.active_adapter], self.lora_C[self.active_adapter],
                         self.lora_D[self.active_adapter],self.lora_B[self.active_adapter],
                         self.scaling[self.active_adapter]).T
            )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        return result


if is_bnb_available():

    class DTSMLinear8bitLt(bnb.nn.Linear8bitLt, DTSMLoraLayer):
        # Low-rank matrix for SVD-based adaptation
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )
            DTSMLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_dropout[self.active_adapter](x) @
                        self.W(self.lora_A[self.active_adapter], self.lora_C[self.active_adapter],
                               self.lora_D[self.active_adapter], self.lora_B[self.active_adapter],
                               self.scaling[self.active_adapter]).T
                    ).to(expected_dtype)
                else:
                    output = (
                        self.lora_dropout[self.active_adapter](x) @
                        self.W(self.lora_A[self.active_adapter], self.lora_C[self.active_adapter],
                               self.lora_D[self.active_adapter],self.lora_B[self.active_adapter],
                               self.scaling[self.active_adapter]).T
                    )
                result = result + output
            return result


if is_bnb_4bit_available():

    class DTSMLinear4bit(bnb.nn.Linear4bit, DTSMLoraLayer):
        # Low-rank matrix for SVD-based adaptation
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            DTSMLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor):
            result = super().forward(x)

            if self.disable_adapters or self.active_adapter not in self.lora_A.keys():
                return result
            elif self.r[self.active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    expected_dtype = result.dtype

                    if x.dtype != torch.float32:
                        x = x.float()
                    output = (
                        self.lora_dropout[self.active_adapter](x) @
                        self.W(self.lora_A[self.active_adapter], self.lora_C[self.active_adapter],
                               self.lora_D[self.active_adapter], self.lora_B[self.active_adapter],
                               self.scaling[self.active_adapter]).T
                    ).to(expected_dtype)
                else:
                    output = (
                        self.lora_dropout[self.active_adapter](x) @
                        self.W(self.lora_A[self.active_adapter], self.lora_C[self.active_adapter],
                               self.lora_D[self.active_adapter], self.lora_B[self.active_adapter],
                               self.scaling[self.active_adapter]).T
                    )
                result = result + output
            return result


