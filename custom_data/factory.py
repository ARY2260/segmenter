import segm.utils.torch as ptu
from custom_data.cityscapes import CityscapesDataset
from segm.data import Loader


def create_dataset(dataset_dir, dataset_kwargs):
    dataset_kwargs = dataset_kwargs.copy()
    dataset_name = dataset_kwargs.pop("dataset")
    batch_size = dataset_kwargs.pop("batch_size")
    num_workers = dataset_kwargs.pop("num_workers")
    split = dataset_kwargs.pop("split")

    # load dataset_name
    if dataset_name == "cityscapes":
        dataset = CityscapesDataset(dataset_dir, split=split, **dataset_kwargs)
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

    dataset = Loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=ptu.distributed,
        split=split,
    )
    return dataset
