import torch
import torch.nn as nn
import numpy as np

from contextlib import nullcontext
from typing import Union, List, Optional, Dict, Any

from snpgen.utils import disabled_train, count_params, expand_dims_like, instantiate_from_config

# Adapted from: https://github.com/Stability-AI/generative-models/blob/main/sgm/modules/encoders/modules.py

class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

from torchinfo import summary


class GeneralConditioner(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

    def __init__(self, emb_models: List[Dict[str, Any]]):
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            # assert isinstance(
            #     embedder, AbstractEmbModel
            # ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(
                    f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}"
                )

            embedders.append(embedder)
        self.embedders = nn.ModuleList(embedders)

    def forward(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None
    ) -> Dict:
        output = dict()
        if force_zero_embeddings is None:
            force_zero_embeddings = []

        for embedder in self.embedders:
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                    emb_out = embedder(batch[embedder.input_key])
                elif hasattr(embedder, "input_keys"):
                    emb_out = embedder(*[batch[k] for k in embedder.input_keys])
            assert isinstance(
                emb_out, (torch.Tensor, list, tuple)
            ), f"encoder outputs must be tensors or a sequence, but got {type(emb_out)}"

            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]

            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                if embedder.ucg_rate > 0.0:
                    emb = (
                        expand_dims_like(
                            torch.bernoulli(
                                (1.0 - embedder.ucg_rate)
                                * torch.ones(emb.shape[0], device=emb.device)
                            ),
                            emb,
                        )
                        * emb
                    )
                if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                    emb = torch.zeros_like(emb)
                    
                if out_key in output:
                    output[out_key] = torch.cat((output[out_key], emb), dim=self.KEY2CATDIM[out_key])
                else:
                    output[out_key] = emb
        return output

    def get_unconditional_conditioning(
        self,
        batch_c: Dict,
        batch_uc: Optional[Dict] = None,
        force_uc_zero_embeddings: Optional[List[str]] = None,
        force_cond_zero_embeddings: Optional[List[str]] = None,
    ):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = list()

        # backup ucg rates and set them to zero
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0

        c = self(batch_c, force_cond_zero_embeddings)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        # restore ucg rates
        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class ClassEmbedder(AbstractEmbModel):
    def __init__(self, embed_dim, n_classes=1000, add_sequence_dim=False):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.add_sequence_dim = add_sequence_dim

    def forward(self, c):
        c = self.embedding(c)
        if self.add_sequence_dim:
            c = c[:, None, :]
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        # Dead code? Conditioning handled in GeneralConditioner
        uc_class = self.n_classes - 1 # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc.long()}
        return uc
