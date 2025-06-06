import os

import torch
import torchvision.datasets as datasets


class PCAM:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=16,
    ):
        
        sub_location = os.path.join(location, "PCAM")

        # self.train_dataset = datasets.PCAM(root=location, download=True, split="train", transform=preprocess)

        # self.train_loader = torch.utils.data.DataLoader(
        #     self.train_dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=num_workers,
        # )

        self.test_dataset = datasets.PCAM(root=sub_location, download=True, split="test", transform=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        self.classnames = [
            "lymph node",
            "lymph node containing metastatic tumor tissue",
        ]
