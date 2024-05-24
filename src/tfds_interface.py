# Copyright 2024 Nikolai Körber. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple PyTorch wrapper to make use of tfds (https://www.tensorflow.org/datasets/tfless_tfds)"""

from torch.utils.data import DataLoader, Dataset
import tensorflow_datasets as tfds
from PIL import Image
from config import ConfigPerco as cfg_perco


class CustomDataset(Dataset):

    def __init__(self, tfds_dataset, transforms):
        self.tfds_dataset = tfds_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.tfds_dataset)

    def __getitem__(self, idx):
        sample = self.tfds_dataset[idx]
        image = Image.fromarray(sample['image'])

        # Ensure all images have 3 channels (RGB)
        image = image.convert('RGB')

        image = self.transforms(image)
        return {'pixel_values': image}


def prepare_dataset(tfds_name, torch_transforms):
    # Load the TensorFlow dataset using tfds.data_source
    ds = tfds.data_source(tfds_name, data_dir=cfg_perco.data_dir)
    # Create PyTorch dataset from tfds.data_source
    return CustomDataset(ds["train"], transforms=torch_transforms)
