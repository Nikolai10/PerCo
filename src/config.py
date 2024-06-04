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

class ConfigPerco:
    # TFDS config
    data_dir = "/content/tensorflow_datasets/"
    # data_dir = "/content/OpenImagesV6"
    # image_list_file = "/content/OpenImagesV6/list_train_files.txt"

    # global path (adjust to your needs)
    global_path = "/content/PerCo/src"

    # BLIP 2 config ({Salesforce/blip2-opt-2.7b, Salesforce/blip2-opt-2.7b-coco, Salesforce/blip2-opt-6.7b, Salesforce/blip2-opt-6.7b-coco})
    # Overview of compatible models: https://huggingface.co/Salesforce
    blip_model = "Salesforce/blip2-opt-2.7b-coco"
    max_number_tokens = 32

    # PerCo
    target_rate = 0.1250

    # see Table 2 in https://arxiv.org/abs/2309.15505 for more information
    # key: target bit-rate
    # value: tuple (x, y), where x and y correspond to the spatial and codebook size, respectively.
    rate_cfg = {}
    rate_cfg[0.1250] = (64, 256)
    rate_cfg[0.0937] = (64, 64)
    rate_cfg[0.0507] = (32, 8196)
    rate_cfg[0.0313] = (32, 256)
    rate_cfg[0.0098] = (16, 1024)
    rate_cfg[0.0024] = (8, 1024)
    rate_cfg[0.0019] = (8, 256)

    # {v_prediction, epsilon}
    prediction_type = "v_prediction"
    # LPIPS loss scalar (if --use_lpips is set)
    lpips_weight = 0.1
    # classifier-free guidance scale
    guidance_scale = 3.0
    # probability of dropping text-conditioning
    cond_drop_prob = 0.1
    # number of sampling steps
    num_inference_steps = 20  # 20 for < 0.05bpp
    # see https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb#scrollTo=9af32168
    random_seed = 3868512668962463
