import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np



class KVIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        K = x
        V = x
        return K, V


class CapInfoNCE(nn.Module):
    def __init__(self, fo=None, fw=None, ku=None, kw=None):
        super().__init__()
        self.fo = fo
        self.fw = fw
        self.ku = ku
        self.kw = kw
        # self.mad_layer = MaxAttDropout()

    def forward(self, o, u, w, mask, return_logits=False):
        """
        Input:
        :o: BoxToxDo contextualized object features
        :u: BoxToxDu uncontextualized object features
        :w: BwxTwxDw caption word features
        :mask: BwxTw word mask
        """
        # assert(o.size()[:1]==w.size()[:1]), 'Bo==Bw'

        Bo, To, Do = o.size()
        _, _, Du = u.size()
        Bw, Tw, Dw = w.size()

        # Compute keys from uncontextualized object features
        Ku = u

        # Compute keys from word features
        Kw = w

        # Compute attention using Ku and Kw
        D = Kw.size(2)
        Kw = Kw.unsqueeze(1).unsqueeze(3)  # Bwx1xTwx1xD
        Ku = Ku.unsqueeze(1)  # Box1xToxD
        att = torch.sum(Kw * Ku, 4, keepdim=True)  # BwxBoxTwxTox1
        att = att / torch.sqrt(torch.tensor(D).float())  # BwxBoxTwxTox1
        att = F.softmax(att, 3)

        # Compute values from contextualized object features
        o = o.unsqueeze(1)  # Box1xToxDo
        V_o = o

        # Compute attended V_o for each word
        att_V_o = torch.sum(att * V_o, 3)  # BwxBoxTwxDo

        w = w.unsqueeze(1)  # Bwx1xTwxDw
        V_w = w  # Bwx1xTwxD

        logits = torch.sum(att_V_o * V_w, 3)  # BwxBoxTw
        log_softmax = F.log_softmax(logits, 1)  # Select image given word
        mask = mask.unsqueeze(1)  # Bwx1xTw
        log_softmax = (1 - mask) * log_softmax
        num_non_mask = torch.sum(1 - mask, 2, keepdim=True)  # Bwx1x1
        log_softmax = log_softmax / (num_non_mask + 1e-6)
        loss = log_softmax.sum(2)
        loss = loss.diag()
        loss = -loss.mean()

        att = att.squeeze(4)

        if return_logits == True:
            return loss, att, att_V_o, logits

        return loss, att, att_V_o


class KVLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.K_layer = nn.Linear(d_in, d_out)
        self.V_layer = nn.Linear(d_in, d_out)

    def forward(self, x):
        B, T, D = x.size()
        x = x.view(-1, D)
        K = self.K_layer(x).view(B, T, -1)
        V = self.V_layer(x).view(B, T, -1)
        return K, V