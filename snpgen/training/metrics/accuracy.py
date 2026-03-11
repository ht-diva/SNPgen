from typing_extensions import Literal
import torch
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics.utilities.compute import _safe_divide, _adjust_weights_safe_divide

class PerClassBinaryAccuracy(BinaryAccuracy):
    def compute(self) -> Tensor:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        return torch.cat([_safe_divide(tn, tn + fp), _safe_divide(tp, tp + fn)], dim=0)


class BalancedBinaryAccuracy(MulticlassAccuracy):
    def __init__(self):
        super().__init__(num_classes=2, average='macro', multidim_average='global')

    # In theory, we could obtain the BalancedBinaryAccuracy by simply using the above __init__ and 
    # not overriding the compute method. However, it won't produce the same results as the 
    # sklearn.metrics.balanced_accuracy_score implementation when y_pred contains classes that are not
    # in y_true, e.g.:
    #     y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    #     y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    #     sklearn.metrics.balanced_accuracy_score(y_true, y_pred) = 0.800
    #     BalancedBinaryAccuracy()(torch.tensor(y_pred), torch.tensor(y_true)) = 0.400
    # This is because MulticlassAccuracy computes the score using _safe_divide, which do not returns NaNs.
    def compute(self) -> Tensor:
        tp, fp, tn, fn = self._final_state()
        score = tp / (tp + fn)

        is_nan = torch.isnan(score)
        score = score[~is_nan]
        tp = tp[~is_nan]
        fp = fp[~is_nan]
        fn = fn[~is_nan]

        return _adjust_weights_safe_divide(score, self.average, False, tp, fp, fn)