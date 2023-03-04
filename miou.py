from segm.metrics import compute_metrics
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu

from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["miou_score"]


@ClassFactory.register(ClassType.GENERAL, alias="miou_score")
def miou_score(test_seg_gt, test_pred_maps, n_cls):
    scores = compute_metrics(
        test_seg_gt,
        test_pred_maps,
        n_cls,
        ignore_index=IGNORE_LABEL,
        ret_cat_iou=True,
        distributed=ptu.distributed,
    )
    mean_accuracy = scores["mean_accuracy"]
    miou_score = scores["mean_iou"]
    print(f"mean accuracy: {mean_accuracy}")
    print(f"miou score: {miou_score}")
    return miou_score
