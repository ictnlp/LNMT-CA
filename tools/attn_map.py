import numpy as np
import torch
import seaborn as sns
import pickle
import argparse
import os
from fairseq import checkpoint_utils, tasks, utils, options
from fairseq.data import encoders
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
#from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn.functional as F
from PIL import Image
import cv2


CKPTS = {
    "mixup-ln": "/home1/yangzhe/experiment/checkpoints/average_epoch.pt",
    # "mtl-ln": "./checkpoints/mustc_ende_stack_base_w2v_6tenc_6dec_pretrain_ex-wmt_mixup-0.0_joint_scale1.0.pt",
    # "baseline-ln": "./checkpoints/mustc_ende_stack_base_w2v_6tenc_6dec_pretrain_ex-wmt.pt"
}
device = "cuda:1"
parser = options.get_generation_parser()
parser.add_argument('--dest', type=str, metavar='N', help='destination')
parser.add_argument('--lang', type=str, metavar='N', help='lang')

args = options.parse_args_and_arch(parser)
print(args)

task = tasks.setup_task(args)
task.load_dataset(args.gen_subset)
tgt_dict = task.target_dictionary
print(list(CKPTS.values()))
models, cfg, _task = checkpoint_utils.load_model_ensemble_and_task(list(CKPTS.values()))

if tgt_dict is None:
    tgt_dict = _task.target_dictionary

print("======= models loaded ========")

def toks2sent(toks):
    _str = tgt_dict.string(toks)
    return _str

def get_hidden_states(model, task):
    model.to(device).eval()
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        # max_sentences=args.batch_size,
        max_sentences=1,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    all_eout = []
    all_eout_token = []
    src_token = []
    for sample in tqdm(itr):
        sample = utils.move_to_cuda(sample,device='cuda:1')
        with torch.no_grad():
            eout, dout = model(**sample["net_input"])
            id = int(sample["id"])
            #print(sample["id"].shape)
            #print(eout.encoder_out.transpose(0, 1).squeeze(0).shape)
            encoder_out = eout.encoder_out.transpose(0, 1)
            encoder_padding_mask = (~eout.encoder_padding_mask).float()
            word_feature = (encoder_out * encoder_padding_mask.unsqueeze(-1)).sum(dim=1) / encoder_padding_mask.sum(
             dim=1).unsqueeze(-1)

            all_eout.append((id, word_feature.cpu().numpy()))
            src_tokens = utils.strip_pad(sample["net_input"]["src_tokens"][0], tgt_dict.pad())
            src_token.append((id, src_tokens))
            all_eout_token.append((id, encoder_out))
            all_eout_token.sort()
            all_eout.sort()
            src_token.sort()


    return all_eout, all_eout_token[8][1],src_token


model_paths = list(CKPTS.values())

print("This scirpt extracts attention matrix from the encoder self-attention layer")
for model, key in zip(models, CKPTS.keys()):
    print(f"===== ckpt: {key} =====")


    all_eout, fr_eout_0, src = get_hidden_states(model, task)

    fr_img_0 = torch.tensor(np.load(f"/home1/yangzhe/test_vit_2.npy"))[8,:,:].float().unsqueeze(0)
    model.to("cpu")
    fr_eout_0 = fr_eout_0.to("cpu")
    _, attn = model.encoder.selective_attns(fr_eout_0.transpose(0,1), fr_img_0.transpose(0,1), fr_img_0.transpose(0,1))
    attn = attn.squeeze(0)[6,:].unsqueeze(0)
    img = Image.open("/home1/yangzhe/image_9.jpg", mode='r')
    plt.figure("/home1/yangzhe/image_9.jpg", figsize=(8, 8))
    attn = attn.view(7, 7).detach().cpu().numpy()
    # Get the word with the dictionary and src_tokens
    # Show the image
    # plt.subplot(math.ceil(attn_map.shape[1] / 4), 4, word_num + 1)
    plt.subplot(1, 1, 1)
    word = src[8][1][6].unsqueeze(0)
    word = toks2sent(word)
    print(word)
    #plt.title(word + '-' + 'origin_word', fontsize=9)
    plt.imshow(img, alpha=1)
    plt.axis('off')

    img_h, img_w = img.size[0], img.size[1]
    attn = cv2.resize(attn.astype(np.float32), (img_h, img_w))
    normed_attn = attn / attn.max()
    normed_attn = (normed_attn * 255).astype('uint8')

    # Show the visual attention map of the word
    plt.imshow(normed_attn, alpha=0.4, interpolation='nearest', cmap='jet')
    plt.axis('off')

    plt.savefig("/home1/yangzhe/test8_2.png",bbox_inches='tight',pad_inches = -0.01)
    plt.show()