import numpy as np
import torch
import torch.nn.functional as F
import ot

if __name__ == '__main__':
    image_feature = np.load("vit-features-pool.npy")
    image_feature = torch.from_numpy(image_feature)
    image_feature = image_feature[0]
    l = torch.nn.Linear(in_features=768, out_features=512)
    image_feature = l(image_feature)

    text_feature = np.load("/home1/yangzhe/fairseq/result2/fr/fr_0.npy")
    text_feature = torch.from_numpy(text_feature).squeeze(0)

    similarity_matrix = 1 - F.cosine_similarity(text_feature.unsqueeze(1), image_feature.unsqueeze(0), dim=2)

    text_feature = text_feature.unsqueeze(0)
    image_feature = image_feature.unsqueeze(0)

    text_pad = torch.tensor(0).repeat(text_feature.size(1)).unsqueeze(0).bool()
    img_pad = torch.tensor(0).repeat(image_feature.size(1)).unsqueeze(0).bool()

    loss = ot.optimal_transport_dist(text_feature, image_feature, text_pad, img_pad)
    print(loss)
