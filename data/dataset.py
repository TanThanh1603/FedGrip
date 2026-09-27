from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Union
import os
import pickle
import torchvision.transforms as transforms
from torchvision.io import read_image
from prefetch_generator import BackgroundGenerator


CURRENT_DIR = Path(__file__).parent.absolute()


class zero_one_normalize:
    def __call__(self, tensor):
        return tensor.float().div(255.0)


class FLDataset(Dataset):
    def __init__(
        self,
        args,
        client_id: Union[int, str],
        device: torch.device = torch.device("cuda"),
    ):
        self.args = args
        self.client_id = client_id
        self.device = device
        client_data_path = os.path.join(
            CURRENT_DIR, args.dataset, args.partition_info_dir, "client_data.pkl"
        )
        with open(client_data_path, "rb") as client_data:
            self.client_data = pickle.load(client_data)
        self.data_paths: List[str] = self.client_data[client_id]["files"]
        self.labels: List[str] = self.client_data[client_id][
            "labels"
        ]  # label for each sample
        dataset_stats = pickle.load(
            open(
                os.path.join(
                    CURRENT_DIR,
                    args.dataset,
                    args.partition_info_dir,
                    "dataset_stats.pkl",
                ),
                "rb",
            )
        )
        self.label_to_index: Dict[int] = {
            label: idx
            for idx, label in enumerate(sorted(dataset_stats["label"].keys()))
        }
        self.num_labels = len(dataset_stats["label"].keys())
        transform = transforms.Compose(
            [
                (
                    transforms.Resize((224, 224))
                    if "domain" not in args.dataset
                    else transforms.Resize((128, 128))
                ),
                zero_one_normalize(),
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        fedgate_weak_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.75, 1.0),
                    ratio=(0.8, 1.25),
                    antialias=True,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                zero_one_normalize(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        augment_transform = transforms.Compose(
            [
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                zero_one_normalize(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if (
            getattr(args, "fedgate_weak_aug", 0) == 1
            and client_id != "test"
            and client_id != "validation"
        ):
            self.transform = fedgate_weak_transform
        elif args.augment and client_id != "test" and client_id != "validation":
            self.transform = augment_transform
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        image_path = self.data_paths[index]
        image = read_image(image_path)
        image = image.to(self.device)
        if image.shape[0] == 1:
            # some images in VLCS dataset are grayscale
            image = image.repeat(3, 1, 1)
        image = self.transform(image)
        label = self.labels[index]
        label = self.label_to_index[label]
        label = torch.tensor(label).to(self.device)
        return image, label


class DataLoaderPrefetch(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
