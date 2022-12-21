import numpy as np
import torch
import torch.nn.functional as F
import json

sum_de_fr = 0
sum_de_cs = 0
sum_fr_cs = 0


def takeSecond(elem):
    return elem[1]


#de-fr
defr = []
decs = []
for i in range(1000):
    defr = []
    decs = []
    de = torch.tensor(np.load(f"~/fairseq/result/de/de_{i}.npy"))
    for j in range(1000):
        fr = torch.tensor(np.load(f"~/fairseq/result/fr/fr_{j}.npy"))
        cs = torch.tensor(np.load(f"~/fairseq/result/cs/cs_{j}.npy"))
        de_fr = F.cosine_similarity(de.unsqueeze(1), fr.unsqueeze(0), dim=2)
        de_cs = F.cosine_similarity(de.unsqueeze(1), cs.unsqueeze(0), dim=2)
        defr.append((j, de_fr))
        decs.append((j, de_cs))

    defr.sort(key=takeSecond, reverse=)
    decs.sort(key=takeSecond)
    for k in range(5):
        if defr[k][0] == i:
            sum_de_fr += 1
            break
    for k in range(5):
        if decs[k][0] == i:
            sum_de_cs += 1
            break

print(sum_de_fr/1000)
print(sum_de_cs/1000)
