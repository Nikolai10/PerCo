# Copyright 2024 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/tensorflow/compression/blob/master/models/ms2020.py,
# Copyright 2020 Google LLC.
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

"""Nonlinear transform coder with Latent Diffusion Model for RGB images.

This is a PyTorch reimplementation of PerCo published in:
M. Careil and M. J. Muckley and J. Verbeek and S. Lathuilière:
"Towards Image Compression with Perfect Realism at Ultra-Low Bitrates"
Int. Conf. on Learning Representations (ICLR), 2024
https://arxiv.org/abs/2310.10325

This script provides compression and decompression functionality.

requires:
- torchac (https://github.com/fab-jul/torchac)
- Ninja

This is meant as 'educational' code - you can use this to get started with your
own experiments.
"""

from config import ConfigPerco as cfg_perco

import sys

sys.path.append(cfg_perco.global_path)

import zlib
import argparse
import sys
import os
import pickle
from absl import app
from absl.flags import argparse_flags
import numpy as np

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage, RandomCrop, CenterCrop
from torchvision import transforms as TR

from transformers import Blip2Processor, Blip2ForConditionalGeneration

from pipeline_sd_perco import StableDiffusionPipelinePerco
from helpers import update_scheduler

import torchac

import bisect

from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)

from cleanfid import fid


def read_png(filename):
    """Loads a PNG image file."""
    img = Image.open(filename).convert('RGB')
    img = TR.functional.resize(img, (512, 512), Image.BICUBIC)
    img_tensor = ToTensor()(img)  # img_tensor in [0, 1]
    return img_tensor


def write_png(filename, image):
    """Saves an PIL image to a PNG file."""
    image.save(filename)


def write_compressed_data_to_file(byte_stream_text, byte_stream_hyper_latent, shape, output_file):
    """Saves compressed representations to file"""
    serialized_text = pickle.dumps(byte_stream_text)
    serialized_hyper_latent = pickle.dumps(byte_stream_hyper_latent)
    serialized_shape = pickle.dumps(shape)
    data_dict = {0: serialized_text, 1: serialized_hyper_latent, 2: serialized_shape}

    with open(output_file, "wb") as fout:
        pickle.dump(data_dict, fout)


def read_compressed_data_from_file(output_file):
    """Loads compressed representations from file"""
    with open(output_file, "rb") as fin:
        data_dict = pickle.load(fin)

    serialized_text = data_dict[0]
    serialized_hyper_latent = data_dict[1]
    serialized_shape = data_dict[2]

    byte_stream_text = pickle.loads(serialized_text)
    byte_stream_hyper_latent = pickle.loads(serialized_hyper_latent)
    shape = pickle.loads(serialized_shape)
    return byte_stream_text, byte_stream_hyper_latent, shape


def compute_cdf_uniform_prob(codebook_size, target_shape):
    """Obtain CDF from uniform distribution, cast to target_shape"""
    b, h, w = target_shape
    prob_per_entry = 1.0 / codebook_size

    # Compute the cumulative sum starting from 0
    cdf = torch.cumsum(torch.full((codebook_size,), prob_per_entry), dim=0)
    cdf = torch.cat([torch.zeros(1), cdf])
    cdf = cdf.view(1, 1, 1, -1).expand(b, h, w, -1)
    cdf = cdf.clone()
    cdf[..., -1] = 1.0
    return cdf


def compress_hyper_latent(z_hat_indices):
    """Compress hyper-latent to bytes using torchac."""
    _, cfg_cs = cfg_perco.rate_cfg[cfg_perco.target_rate]
    cdf = compute_cdf_uniform_prob(cfg_cs, z_hat_indices.shape)
    z_hat_indices = z_hat_indices.to(torch.int16).to('cpu')
    return torchac.encode_float_cdf(cdf, z_hat_indices, check_input_bounds=True)


def decompress_hyper_latent(compressed_hyper_latent, shape):
    """Decompress hyper-latent using torchac."""
    cfg_ss, cfg_cs = cfg_perco.rate_cfg[cfg_perco.target_rate]
    H, W = shape
    factor = 512 // cfg_ss
    h, w = H // factor, W // factor
    cdf = compute_cdf_uniform_prob(cfg_cs, (1, int(h), int(w)))
    return torchac.decode_float_cdf(cdf, compressed_hyper_latent)


def compress_text(input_text):
    """Compress the input text to bytes using zlib."""
    input_bytes = input_text.encode('utf-8')
    return zlib.compress(input_bytes, level=zlib.Z_BEST_COMPRESSION)


def decompress_text(compressed_text):
    """Decompress the compressed text using zlib."""
    decompressed_bytes = zlib.decompress(compressed_text)
    return decompressed_bytes.decode('utf-8')


def calculate_bpp(compressed_data, num_pixels, bytes=True, num_bytes=None):
    """Calculate bpp given the compressed text and number of pixels."""
    scaling_factor = 8 if bytes else 1
    if num_bytes:
        return num_bytes * scaling_factor / num_pixels
    return len(compressed_data) * scaling_factor / num_pixels


def compress(args):
    """Compresses an image."""

    # --- ENCODER ---
    # 1. Read image
    # 2. Get image caption (BLIP 2)
    # 3. Compress caption (zlib)
    # 4. Run VAE encoder
    # 5. Run hyper-encoder
    # 6. Compress hyper-latent (AC)
    # 7. Write compressed representations to file for further processing
    # 8. Optional: measure performance
    print('\n--- Start Compression ---\n')

    # 1. Read image
    print('Reading image {}...'.format(args.input_file))
    img = read_png(args.input_file)
    H, W = img.shape[1], img.shape[2]

    # 2. Get image caption (BLIP 2)
    print('Retrieving {} caption...'.format(cfg_perco.blip_model))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(cfg_perco.blip_model)
    blip2 = Blip2ForConditionalGeneration.from_pretrained(cfg_perco.blip_model).to(device)
    inputs = processor(images=img * 255, return_tensors="pt").to(device)
    generated_ids = blip2.generate(**inputs, max_length=cfg_perco.max_number_tokens)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print('Generated caption: {}'.format(generated_text))

    # 3. Compress caption (zlib)
    print('Compressing caption using zlib...')
    byte_stream_text = compress_text(generated_text)
    print('Compressed text (bytes): {}'.format(byte_stream_text))
    bpp_text = calculate_bpp(byte_stream_text, H * W)
    print('BPP text: {:.4f}'.format(bpp_text))

    # 4. Run VAE encoder
    # (here we make use of the convenient pipeline functionality; should be simplified later on)
    print('Running VAE encoder...')
    pipe = StableDiffusionPipelinePerco.from_pretrained(args.model_path,
                                                        safety_checker=None,
                                                        requires_safety_checker=False)
    # update_scheduler(pipe)
    pipe.to(device)

    latents = pipe.vae.encode((img * 2. - 1.).unsqueeze(0).to(device)).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    print('Encoded image shape: {}'.format(latents.shape))

    # 5. Run hyper-encoder
    print('Running hyper-encoder...')
    pipe.hyper_enc.quantizer.eval()
    hyper_latent = pipe.hyper_enc(latents)
    z_hat, z_hat_indices = hyper_latent.z_hat, hyper_latent.indices
    print('Encoded hyper-latent shape: {}'.format(z_hat.shape))

    # 6. Compress hyper-latent (AC)
    print('Compressing hyper-latent...')
    byte_stream_hyper_latent = compress_hyper_latent(z_hat_indices)
    bpp_hyper_latent = calculate_bpp(byte_stream_hyper_latent, H * W)
    print('BPP hyper-latent: {:.4f}'.format(bpp_hyper_latent))

    # 7. Write compressed representations to file for further processing
    print('Writing compressed representations to file...')
    # h, w = z_hat.shape[2], z_hat.shape[3]
    write_compressed_data_to_file(byte_stream_text, byte_stream_hyper_latent, (H, W), args.output_file)
    print('BPP text + hyper-latent: {:.4f}'.format(bpp_text + bpp_hyper_latent))
    bpp_total = calculate_bpp(None, H * W, num_bytes=os.path.getsize(args.output_file))
    print('BPP total (including pickle overhead): {:.4f}'.format(bpp_total))

    # 8. Optional: measure performance
    if args.verbose:
        print('Measuring performance...')
        # Read from a file.
        byte_stream_text_read, byte_stream_hyper_latent_read, shape_read = read_compressed_data_from_file(
            args.output_file)
        z_hat_indices_restored = decompress_hyper_latent(byte_stream_hyper_latent_read, shape_read).to(device)
        generated_text_restored = decompress_text(byte_stream_text_read)

        assert z_hat_indices_restored.equal(z_hat_indices)
        assert generated_text_restored == generated_text

        pipe = StableDiffusionPipelinePerco.from_pretrained(args.model_path,
                                                            safety_checker=None,
                                                            requires_safety_checker=False)
        # update_scheduler(pipe)
        pipe.to(device)

        # pipe.hyper_enc.quantizer.eval()
        z_hat_restored = pipe.hyper_enc.quantizer.get_output_from_indices(z_hat_indices_restored.long())
        z_hat_restored = z_hat_restored.permute(0, 3, 1, 2)

        assert torch.all(z_hat == z_hat_restored)

        generator = torch.Generator(device=device)
        generator.manual_seed(cfg_perco.random_seed)
        x_hat = pipe(generated_text_restored,
                     z_hat_restored,
                     height=H,
                     width=W,
                     num_inference_steps=cfg_perco.num_inference_steps,
                     guidance_scale=cfg_perco.guidance_scale,
                     generator=generator).images[0]

        # Cast to float in order to compute metrics.
        x = img
        x_hat = ToTensor()(x_hat)

        mse = F.mse_loss(x, x_hat)
        psnr = -10. * torch.log10(mse)

        print(f"Mean squared error: {mse.item():0.4f}")
        print(f"PSNR (dB): {psnr.item():0.2f}")
        print(f"Bits per pixel: {bpp_total:0.4f}")

    print('\n--- End Compression ---\n')


def decompress(args):
    """Decompresses an image."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # --- Decoder ---
    # 1. Decompress caption (zlib), hyper-latent (AC)
    # 2. Run decompression (handled by our custom StableDiffusionPipelinePerco pipeline)
    # 3. Write reconstruction to file for further processing
    print('\n--- Start Decompression ---\n')

    # 1. Decompress caption (zlib), hyper-latent (AC)
    print('Decompressing hyper-latent, caption...')
    byte_stream_text_read, byte_stream_hyper_latent_read, shape_read = read_compressed_data_from_file(args.input_file)
    z_hat_indices_restored = decompress_hyper_latent(byte_stream_hyper_latent_read, shape_read).to(device)
    generated_text_restored = decompress_text(byte_stream_text_read)

    # 2. Run decompression (handled by our custom StableDiffusionPipelinePerco pipeline)
    print('Restoring image...')
    print('Using classifier-free guidance with scale: {}'.format(cfg_perco.guidance_scale))
    print('Using {} sampling steps.'.format(cfg_perco.num_inference_steps))

    H, W = shape_read
    pipe = StableDiffusionPipelinePerco.from_pretrained(args.model_path,
                                                        safety_checker=None,
                                                        requires_safety_checker=False)
    # update_scheduler(pipe)
    pipe.to(device)

    pipe.hyper_enc.quantizer.eval()
    z_hat_restored = pipe.hyper_enc.quantizer.get_output_from_indices(z_hat_indices_restored.long())
    z_hat_restored = z_hat_restored.permute(0, 3, 1, 2)

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg_perco.random_seed)
    x_hat = pipe(generated_text_restored,
                 z_hat_restored,
                 height=H,
                 width=W,
                 num_inference_steps=cfg_perco.num_inference_steps,
                 guidance_scale=cfg_perco.guidance_scale,
                 generator=generator).images[0]
    print('Reconstructed image shape: {}'.format(x_hat))

    # 3. Write reconstruction to file for further processing
    print('Save reconstruction to file...')
    write_png(args.output_file, x_hat)

    print('\n--- End Decompression ---\n')


def eval_trained_model_dist(out_dir):
    """compute FID, KID based on previously extracted patches"""

    out_dir_real = os.path.join(out_dir + 'patches_real/')
    out_dir_fake = os.path.join(out_dir + 'patches_fake/')
    out_file = os.path.join(out_dir + 'results.txt')

    # compute FID, KID
    print('\ncomputing FID, KID...')
    fid_score = fid.compute_fid(out_dir_real, out_dir_fake)
    kid_score = fid.compute_kid(out_dir_real, out_dir_fake)
    print(f'\nFID-score: {fid_score}\nKID-score: {kid_score}')
    with open(out_file, "a") as fo:
        fo.write(f'\nFID-score: {fid_score}\nKID-score: {kid_score}')
    print('Done!')


def evaluate_ds(args):
    """evaluate whole dataset"""

    in_dir = args.in_dir
    out_dir = args.out_dir

    out_dir_real = os.path.join(out_dir + 'patches_real/')
    out_dir_fake = os.path.join(out_dir + 'patches_fake/')
    if not os.path.exists(out_dir_real):
        os.makedirs(out_dir_real)
    if not os.path.exists(out_dir_fake):
        os.makedirs(out_dir_fake)

    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # prepare BLIP 2
    processor = Blip2Processor.from_pretrained(cfg_perco.blip_model)
    blip2 = Blip2ForConditionalGeneration.from_pretrained(cfg_perco.blip_model).to(device)

    # prepare StableDiffusionPipelinePerco
    pipe = StableDiffusionPipelinePerco.from_pretrained(args.model_path,
                                                        safety_checker=None,
                                                        requires_safety_checker=False)
    # update_scheduler(pipe)
    pipe.to(device)

    # prepare metrics
    msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=False)

    to_pil = ToPILImage()

    bpp_text_arr = []
    bpp_hyper_latent_arr = []
    bpp_total_arr = []
    psnr_arr = []
    msssim_arr = []
    lpips_arr = []

    results_file = os.path.join(out_dir, "results.txt")
    filenames = sorted(os.listdir(in_dir))

    # continue on failure
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as file:
            # obtain last processed filename
            lines = file.readlines()
            last_line = lines[-1].strip()
            filename = last_line.split(',')[0]

            # restore previously calculated values
            for line in lines:
                bpp_text, bpp_hyper_latent, bpp_total, psnr, msssim, lpips = line.strip().split(',')[-6:]
                bpp_text_arr.append(float(bpp_text))
                bpp_hyper_latent_arr.append(float(bpp_hyper_latent))
                bpp_total_arr.append(float(bpp_total))
                psnr_arr.append(float(psnr))
                msssim_arr.append(float(msssim))
                lpips_arr.append(float(lpips))

        # retrieve open subset
        old_length = len(filenames)
        start_index = bisect.bisect_left(filenames, filename + '.jpg')
        filenames = filenames[start_index + 1:]
        new_length = len(filenames)

        # add assertion
        assert old_length - new_length == len(bpp_text_arr)

    # Iterate over each image in the dataset
    for filename in filenames:
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(in_dir, filename)
            print(f"Processing {image_path}")

            # 1. Read image
            img = read_png(image_path)
            H, W = img.shape[1], img.shape[2]

            # 2. Get image caption (BLIP 2)
            inputs = processor(images=img * 255, return_tensors="pt").to(device)
            generated_ids = blip2.generate(**inputs, max_length=cfg_perco.max_number_tokens)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            # 3. Compress caption (zlib) --> just to measure bpp (text)
            byte_stream_text = compress_text(generated_text)
            bpp_text = calculate_bpp(byte_stream_text, H * W)

            # 4. Run VAE encoder
            latents = pipe.vae.encode((img * 2. - 1.).unsqueeze(0).to(device)).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

            # 5. Run hyper-encoder
            pipe.hyper_enc.quantizer.eval()
            hyper_latent = pipe.hyper_enc(latents)
            z_hat, z_hat_indices = hyper_latent.z_hat, hyper_latent.indices

            # 6. Compress hyper-latent (AC) --> just to measure bpp (hyper-latent)
            byte_stream_hyper_latent = compress_hyper_latent(z_hat_indices)
            bpp_hyper_latent = calculate_bpp(byte_stream_hyper_latent, H * W)
            bpp_total = bpp_text + bpp_hyper_latent

            # 7. Generate reconstruction (skip compress -> decompress for speed reasons, logic remains correct)
            # generator = torch.Generator(device=device)
            # generator.manual_seed(cfg_perco.random_seed)
            generator = None
            x_hat = pipe(generated_text,
                         z_hat,
                         height=H,
                         width=W,
                         num_inference_steps=cfg_perco.num_inference_steps,
                         guidance_scale=cfg_perco.guidance_scale,
                         generator=generator).images[0]

            # 8. Write files + metadata to disk
            filename_wo_ext = os.path.splitext(os.path.basename(image_path))[0]
            out_path_orig = out_dir_real + filename_wo_ext + '_inp.png'
            out_path_rec = out_dir_fake + filename_wo_ext + '_otp.png'

            write_png(out_path_rec, x_hat)
            write_png(out_path_orig, to_pil(img))

            # 9. Compute metrics
            x = img
            x_hat = ToTensor()(x_hat)

            mse = F.mse_loss(x, x_hat)
            psnr = -10. * torch.log10(mse)
            msssim = msssim_metric(x.unsqueeze(0), x_hat.unsqueeze(0))
            lpips = lpips_metric(x.unsqueeze(0) * 2 - 1., x_hat.unsqueeze(0) * 2 - 1.)

            with open(results_file, "a", encoding="utf-8") as text_file:
                text_file.write(
                    '{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(filename_wo_ext, generated_text,
                                                                               bpp_text, bpp_hyper_latent, bpp_total,
                                                                               psnr, msssim, lpips))

            bpp_text_arr.append(bpp_text)
            bpp_hyper_latent_arr.append(bpp_hyper_latent)
            bpp_total_arr.append(bpp_total)
            psnr_arr.append(psnr)
            msssim_arr.append(msssim)
            lpips_arr.append(lpips)

    # 10. Write summary to disk
    with open(results_file, "a") as text_file:
        text_file.write('\n{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}'.format('avg', np.mean(bpp_text_arr),
                                                                                np.mean(bpp_hyper_latent_arr),
                                                                                np.mean(bpp_total_arr),
                                                                                np.mean(psnr_arr), np.mean(msssim_arr),
                                                                                np.mean(lpips_arr)))

    print('Done!')


def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # High-level options.
    parser.add_argument(
        "--verbose", "-V", action="store_true",
        help="Report progress and metrics when compressing.")
    parser.add_argument(
        "--model_path", default="cmvl_2024",
        help="Path where to save/load the trained model.")
    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="What to do: 'compress' reads an image file (lossless "
             "PNG format) and writes a compressed pickle file. 'decompress' "
             "reads a pickle file and reconstructs the image (in PNG format). "
             "input and output filenames need to be provided for the latter "
             "two options. Invoke '<command> -h' for more information.")

    # 'evaluate_ds' subcommand.
    evaluate_ds_cmd = subparsers.add_parser(
        "evaluate_ds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="evaluates the compression performance on a whole dataset")
    evaluate_ds_cmd.add_argument("--in_dir", help="Input directory")
    evaluate_ds_cmd.add_argument('--out_dir', required=True, help='Where to save outputs.')
    evaluate_ds_cmd.add_argument('--mode', type=int, default=0, help=(
        '0 : BPP+PSNR+LPIPS+MS-SSIM; 1 : FID+KID; always run mode 0 before mode 1 to extract image patches!'))

    # 'compress' subcommand.
    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a PNG file, compresses it, and writes as pickle file.")

    # 'decompress' subcommand.
    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a pickle file, reconstructs the image, and writes back "
                    "a PNG file.")

    # Arguments for both 'compress' and 'decompress'.
    for cmd, ext in ((compress_cmd, ".pkl"), (decompress_cmd, ".png")):
        cmd.add_argument(
            "input_file",
            help="Input filename.")
        cmd.add_argument(
            "output_file", nargs="?",
            help=f"Output filename (optional). If not provided, appends '{ext}' to "
                 f"the input filename.")

    # Parse arguments.
    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args


def main(args):
    if args.command == "compress":
        if not args.output_file:
            args.output_file = args.input_file + ".pkl"
        compress(args)
    elif args.command == "decompress":
        if not args.output_file:
            args.output_file = args.input_file + ".png"
        decompress(args)
    elif args.command == "evaluate_ds":
        if args.mode == 0:
            evaluate_ds(args)
        else:
            eval_trained_model_dist(args.out_dir)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
