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
            src_sents = toks2sent(src_tokens)
            src_token.append((id,src_sents))
            all_eout_token.append((id, encoder_out))
            all_eout_token.sort()
            all_eout.sort()
            src_token.sort()


    return all_eout, all_eout_token[0][1],src_token


model_paths = list(CKPTS.values())

print("This scirpt extracts attention matrix from the encoder self-attention layer")
for model, key in zip(models, CKPTS.keys()):
    print(f"===== ckpt: {key} =====")


    all_eout, fr_eout_0, src = get_hidden_states(model, task)

    X = all_eout[0][1]
    for i in range(len(all_eout)-1):
        X = np.vstack((X, all_eout[i+1][1]))

    ts = TSNE(n_components=2)
    pca = PCA(n_components=2)
    np.save("/home1/yangzhe/fr-con-token.npy", X)



    a=np.load("/home1/yangzhe/fr-con.npy")
    b=np.load("/home1/yangzhe/cs-nocon.npy")
    c=np.load("/home1/yangzhe/fr-con-token.npy")
    c1=np.load("/home1/yangzhe/de-t.npy")
    d=np.load("/home1/yangzhe/test1_vit.npy")

    d=torch.tensor(d).float()

    model.to("cpu")
    d = model.encoder.normlayer1(d)
    '''
    fr_img_0 = torch.tensor(np.load(f"/home1/yangzhe/fr_test_1.npy")).float()

    fr_eout_0 = fr_eout_0.to("cpu")
    _, attn = model.encoder.selective_attns(fr_eout_0.transpose(0,1), fr_img_0.transpose(0,1), fr_img_0.transpose(0,1))
    attn = attn.squeeze(0)[6,:].unsqueeze(0)
    img = Image.open("/home1/yangzhe/1007129816.jpg", mode='r')
    plt.figure("/home1/yangzhe/1007129816.jpg", figsize=(8, 8))
    attn = attn.view(7, 7).detach().cpu().numpy()
    # Get the word with the dictionary and src_tokens
    # Show the image
    # plt.subplot(math.ceil(attn_map.shape[1] / 4), 4, word_num + 1)
    plt.subplot(1, 1, 1)
    plt.title('word' + '-' + 'origin_word', fontsize=9)
    plt.imshow(img, alpha=1)
    plt.axis('off')

    img_h, img_w = img.size[0], img.size[1]
    attn = cv2.resize(attn.astype(np.float32), (img_h, img_w))
    normed_attn = attn / attn.max()
    normed_attn = (normed_attn * 255).astype('uint8')

    # Show the visual attention map of the word
    plt.imshow(normed_attn, alpha=0.4, interpolation='nearest', cmap='jet')
    plt.axis('off')
    plt.show()
    '''

    d=d.detach().numpy()

    d = torch.tensor(d).float().to("cuda:0")
    a = torch.tensor(c).float().to("cuda:0")

    i_norm = F.normalize(d, p=2, dim=-1, eps=1e-5)
    w_norm = F.normalize(a, p=2, dim=-1, eps=1e-5)
    similarity_matrix = w_norm.matmul(i_norm.transpose(0, 1))
    similarity_matrix = similarity_matrix.to("cuda:0")
    sim_exp = torch.exp(similarity_matrix / 0.007)
    count = 0
    for i in range(1000):
        #index = torch.argmax(sim_exp[i])
        a, idx1 = torch.sort(sim_exp[i], descending=True)
        index = idx1[:3]
        if i in index:
            count += 1
    acc = count/1000

    d = d.cpu().detach().numpy()
    wi = np.vstack((c,d))
    #wi = np.vstack((wi,c))

    #c=np.load("/home1/yangzhe/test_2016_flickr-resnet50-avgpool.npy")
    wi = ts.fit_transform(wi)

    data0 = pd.DataFrame(wi[0:1000,:], columns=['x', 'y'])  # 将data,data2封装成DataFrame的x,y列，方便后面进行联合分布
    data0['lan'] = 'fr'
    #data0['index'] = [0,1,2,3,4,5,6,7,8,9]
    data1 = pd.DataFrame(wi[1000:2000,:], columns=['x', 'y'])  # 将data,data2封装成DataFrame的x,y列，方便后面进行联合分布
    data1['lan'] = 'i'
    #data2 = pd.DataFrame(wi[2000:3000, :], columns=['x', 'y'])  # 将data,data2封装成DataFrame的x,y列，方便后面进行联合分布
    #data2['lan'] = 'de'
    #data2 = pd.DataFrame(c, columns=['x', 'y'])  # 将data,data2封装成DataFrame的x,y列，方便后面进行联合分布
    #data2['lan'] = 'de'
    #data1['index'] = [0,1,2,3,4,5,6,7,8,9]
    data = pd.concat([data0, data1], axis=0)
    #data = pd.concat([data, data2], axis=0)

    #data = pd.concat([data, data2], axis=0)
    data.reset_index(drop=True, inplace=True)


    # penguins = sns.load_dataset("penguins")
    sns.jointplot(data=data, x="x", y="y", hue="lan")
    plt.show()



    with open(f"{args.dest}/{args.lang}_text.txt", "w") as fout:
        for src in all_src_sents:
            fout.write(str(src[1]) + "\n")

    for src in all_eout:
        np.save(f"{args.dest}/{args.lang}/{args.lang}_{src[0]}.npy", src[1])
