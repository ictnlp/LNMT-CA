import numpy as np
import seaborn as sns
import torch
from torch.nn import functional as F
import argparse
import os
from fairseq import checkpoint_utils, tasks, utils, options
#from fairseq.data import encoders
from tqdm import tqdm
#from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import pandas as pd
import matplotlib.pyplot as plt


CKPTS = {
    "mixup-ln": "/home1/yangzhe/experiment/checkpoints/multilingual-test/average_epoch.pt",
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

def cost_matrix_cosine(x, y, eps=1e-5):
    """ Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist

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
    all_src_sents = []
    x_axis = []
    y_axis = []
    for sample in tqdm(itr):
        sample = utils.move_to_cuda(sample,device='cuda:1')
        with torch.no_grad():
            eout, dout = model(**sample["net_input"])
            id = int(sample["id"])
            #print(sample["id"].shape)
            #print(eout.encoder_out.transpose(0, 1).squeeze(0).shape)
            encoder_out = eout.encoder_out.transpose(0, 1)
            encoder_padding_mask = ~eout.encoder_padding_mask
            word_feature = (encoder_out * encoder_padding_mask.unsqueeze(-1))

            img_pad = torch.tensor(np.load("test_pads.npy"))
            img_feature = torch.tensor(np.load("test_results.npy"))
            img_classes = torch.tensor(np.load("test_names.npy"))

            img_feature.to(device)

            x = model.encoder.linear1(img_feature)
            residual = x.to(device)

            x = model.encoder.fc3(x)
            x = model.encoder.relu(x)
            x = model.encoder.fc4(x)

            x = residual + x

            x = model.encoder.normlayer2(x)

            cost_matrix = cost_matrix_cosine(word_feature.unsqueeze(0), x[0].unsqueeze(0))

            with open("classes.txt", 'r') as f:
                classes = f.readlines()

            classes.append("<pad>")
            for i in range(img_classes.shape[1]):
                x_axis.append(classes[img_classes[0][i]])

            src_text = list(toks2sent(sample["net_input"]["src_tokens"]))

            for i in range(encoder_padding_mask.shape[1]):
                if encoder_padding_mask[i] == True:
                    y_axis.append(src_text[i])
                else:
                    y_axis.append("<pad>")


            df = pd.DataFrame(cost_matrix[0], index=y_axis, columns=x_axis)
            sns.heatmap(df, annot=True)
            plt.savefig("./confusion_matrix.jpg")
            plt.show()

    return all_eout, all_src_sents


model_paths = list(CKPTS.values())

print("This scirpt extracts attention matrix from the encoder self-attention layer")
for model, key in zip(models, CKPTS.keys()):
    print(f"===== ckpt: {key} =====")
    all_eout, all_src_sents = get_hidden_states(model, task)

