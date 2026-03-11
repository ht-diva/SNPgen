from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.utilities.compute import _safe_divide

from torchmetrics.functional.classification.stat_scores import (
    _multiclass_stat_scores_format,
    _multiclass_stat_scores_tensor_validation,
    _multiclass_stat_scores_update,
)

class ReconstructionAccuracy(MulticlassAccuracy):
    def __init__(self, average='micro', **kwargs):
        super().__init__(average=average, **kwargs)

    def update(self, preds: Tensor, target: Tensor, keep_indices=None) -> None:
        """Update state with predictions and targets."""
        if keep_indices is not None:
            preds = preds[keep_indices]
            target = target[keep_indices]

        if self.validate_args:
            _multiclass_stat_scores_tensor_validation(
                preds, target, self.num_classes, self.multidim_average, self.ignore_index
            )
        preds, target = _multiclass_stat_scores_format(preds, target, self.top_k)
        tp, fp, tn, fn = _multiclass_stat_scores_update(
            preds, target, self.num_classes, self.top_k, self.average, self.multidim_average, self.ignore_index
        )
        self._update_state(tp, fp, tn, fn)
    


# class ReconstructionAccuracy(Metric):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, preds: Tensor, target: Tensor) -> None:
#         preds, target = self._input_format(preds, target)
#         if preds.shape != target.shape:
#             raise ValueError("preds and target must have the same shape")

#         self.correct += torch.sum(preds == target)
#         self.total += target.numel()

#     def compute(self) -> Tensor:
#         return self.correct.float() / self.total