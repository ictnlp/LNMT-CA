import timm
from torchvision import datasets
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

if __name__ == '__main__':

    device = 'cuda'

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = datasets.ImageFolder('~/multi30k-dataset/dataset/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=False, num_workers=4)

    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.to(device)
    model.eval()

    img_feature = torch.empty(10, 196, 768)
    tmp = torch.empty(10, 196, 768)

    pool = torch.nn.AvgPool2d(kernel_size=(14, 1))
    for i, (img, label) in enumerate(train_loader):
        img = img.to(device)
        with torch.no_grad():
            image_embs = pool(model.forward_features(img)).float().to("cpu")
            #print(image_embs.shape)
            if i == 0:
                img_feature = image_embs
            else:
                tmp = image_embs
                img_feature = torch.cat((img_feature, tmp), 0)

        if (i+1) % 10 == 0:
            print("finish loading {}".format((i + 1) * 100))
    print(img_feature.shape)
    img_feature = img_feature.numpy()

    np.save("vit-features-pool.npy", img_feature)
