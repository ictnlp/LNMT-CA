import urllib
from PIL import Image
import torch
import argparse
from os.path import join
import numpy as np
import clip
from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser(description="Vit")
    parser.add_argument("--lang")
    parser.add_argument("--device")

    return parser


def main():
    args = get_parser().parse_args()
    device = args.device
    model, preprocess = clip.load("ViT-B/32", device=device)

    print("loding")
    images_path = f'../data/img/raw_images'
    image_splits = f'../data/img/{args.lang}_img_name.txt'

    with open(image_splits, 'r') as f:
        image_list = list(map(str.strip, f.readlines()))

    results = []
    results2 = []

    with torch.no_grad():
        for idx in range(0, len(image_list)):
            print('image: {0}/{1}'.format(idx, len(image_list)))
            image_path = join(images_path, image_list[idx])

            img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            out, out1 = model.encode_image(img)
            results.append(out.squeeze(0).cpu().numpy().tolist())
            out1 = out1.cpu().numpy()
            np.save(f"../data/img/{args.lang}/{args.lang}_vit_clip_{idx}",out1)

    tmp_np_results = np.array(results)
    print(tmp_np_results.shape)
    np.save(f"../data/img/{args.lang}/{args.lang}_vit_clip_avg", tmp_np_results)
    
if __name__ == '__main__':
    main()
