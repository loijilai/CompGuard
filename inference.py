import torch
from torchvision import transforms
import argparse
from PIL import Image
import math
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
from compressai.zoo import bmshj2018_factorized

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
    input_img = 'input.png'
    output_img = 'output.png'
    parser = argparse.ArgumentParser(description='Compress images using learned image compression models')
    parser.add_argument('--input', type=str, default=input_img, help='input image')
    parser.add_argument('--output', type=str, default=output_img, help='output image')
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
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    net = bmshj2018_factorized(quality=2, pretrained=True).eval().to(device)     
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        out_net = net.forward(x)
        out_net['x_hat'].clamp_(0, 1)

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