import torch
from torchvision import transforms
import argparse
from PIL import Image
import math
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
from compressai.zoo import bmshj2018_factorized
from compressai.zoo import image_models

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def parse_args():
    input_img = '/tmp2/loijilai/compguard/dataset/train/frame_0.bmp'
    output_img = '/tmp2/loijilai/compguard/outputs/output.png'
    checkpoint = None
    parser = argparse.ArgumentParser(description='Compress images using learned image compression models')

    # Model
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument('--quality', type=int, default=2, help='compression quality')
    parser.add_argument('--checkpoint', type=str, default=checkpoint, help='path to the model checkpoint')

    # Image
    parser.add_argument('--input', type=str, default=input_img, help='input image')
    parser.add_argument('--output', type=str, default=output_img, help='output image')
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=2,
        default=(512, 512),
        help="Size of the patches to be cropped (default: %(default)s)",
    )

    # Inference setting
    parser.add_argument('--recompress', type=int, default=1, help='number of recompression iterations')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print("WARNING: CUDA is not available. Running on CPU.")
        device = 'cpu'


    img = Image.open(args.input).convert('RGB')

    transformation = transforms.Compose([
        transforms.CenterCrop(args.patch_size),  # Crop the image to the specified size at the center
        transforms.ToTensor()  # Converts the PIL image or a numpy.ndarray (HxWxC) in the range [0, 255] to a floating-point PyTorch tensor of shape (CxHxW) in the range [0.0, 1.0]
    ])

    x = transformation(img).unsqueeze(0).to(device)

    if(args.checkpoint is None):
        # net = bmshj2018_factorized(quality=args.quality, pretrained=True).eval().to(device)     
        net = image_models[args.model](quality=args.quality, pretrained=True).eval().to(device)
    else:
        # net = bmshj2018_factorized(quality=args.quality, pretrained=False).eval().to(device)     
        net = image_models[args.model](quality=args.quality, pretrained=False).eval().to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_ = x
    with torch.no_grad():
        for i in range(args.recompress):
            print(f'Iteration {i+1}/{args.recompress}')
            out_net = net.forward(x_)
            out_net['x_hat'].clamp_(0, 1)
            x_ = out_net['x_hat']

    rec_net = transforms.ToPILImage()(out_net['x_hat'].squeeze().cpu()) 
    diff = torch.mean((out_net['x_hat'] - x).abs(), axis=1).squeeze().cpu()
    fix, axes = plt.subplots(1, 3, figsize=(16, 12))

    for ax in axes:
        ax.axis('off')
        
    axes[0].imshow(img)
    axes[0].title.set_text('Original')

    axes[1].imshow(rec_net)
    axes[1].title.set_text('Reconstructed')

    axes[2].imshow(diff, cmap='viridis')
    axes[2].title.set_text('Difference')

    plt.savefig(args.output, bbox_inches='tight')

    print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
    print(f'MS-SSIM: {compute_msssim(x, out_net["x_hat"]):.4f}')
    print(f'Bit-rate: {compute_bpp(out_net):.3f} bpp')

if __name__ == '__main__':
    main()