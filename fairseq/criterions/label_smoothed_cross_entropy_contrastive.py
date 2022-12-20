# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion



def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("label_smoothed_cross_entropy_contrastive")
class LabelSmoothedCrossEntropyCriterionContrastive(FairseqCriterion):
    def __init__(
        self,
        task,
        image_root,
        sentence_level,
        token_level,
        de_sen,
        fr_sen,
        cs_sen,
        sen_tem,
        token_tem,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.root = image_root
        self.sentence_level = sentence_level
        self.token_level = token_level
        self.de_sen = de_sen
        self.fr_sen = fr_sen
        self.cs_sen = cs_sen
        self.sen_tem = sen_tem
        self.token_tem = token_tem

        self.de_image_matrix = torch.tensor(np.load(f"{self.root}/de_vit_clip_avg.npy")).float()
        self.fr_image_matrix = torch.tensor(np.load(f"{self.root}/fr_vit_clip_avg.npy")).float()
        self.cs_image_matrix = torch.tensor(np.load(f"{self.root}/cs_vit_clip_avg.npy")).float()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--image_root', default=0, type=str,
                            help='The root of your image tensor')
        parser.add_argument('--sentence_level', default=True, type=bool,
                            help='Sentence-level contrastive loss')
        parser.add_argument('--token_level', default=False, type=bool,
                            help='Token-level contrastive loss')
        parser.add_argument('--de_sen', default=60136, type=int,
                            help='Number of de sentences')
        parser.add_argument('--fr_sen', default=60136, type=int,
                            help='Number of fr sentences')
        parser.add_argument('--cs_sen', default=60136, type=int,
                            help='Number of cs sentences')
        parser.add_argument('--sen_tem', default=0.01, type=float,
                            help='Temperature of sentence-level contrastive learning')
        parser.add_argument('--token_tem', default=0.1, type=float,
                            help='Temperature of token-level contrastive learning')
        # fmt: on


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        eout, net_output = model(**sample["net_input"])
        contrastive_loss = 0
        token_loss = 0
        token_loss_de = 0
        token_loss_fr = 0
        token_loss_cs = 0
        token_loss_en = 0

        contrastive_loss_de = 0
        contrastive_loss_fr = 0
        contrastive_loss_cs = 0
        contrastive_loss_en = 0
        first = True

        if self.training:
            de_sample, de_e_out, de_output = self.find_lang(sample, model, "de")
            if de_sample != 0:
                if self.sentence_level:
                    contrastive_loss_de = self.compute_conloss(de_sample, de_e_out, model, "de")
                if self.token_level:
                    token_loss_de = self.compute_token_loss(de_sample, de_e_out, model, "de")

            fr_sample, fr_e_out, fr_output = self.find_lang(sample, model, "fr")
            if fr_sample != 0:
                if self.sentence_level:
                    contrastive_loss_fr = self.compute_conloss(fr_sample, fr_e_out, model, "fr")
                if self.token_level:
                    token_loss_fr = self.compute_token_loss(fr_sample, fr_e_out, model, "fr")

            cs_sample, cs_e_out, cs_output = self.find_lang(sample, model, "cs")
            if cs_sample != 0:
                if self.sentence_level:
                    contrastive_loss_cs = self.compute_conloss(cs_sample, cs_e_out, model, "cs")
                if self.token_level:
                    token_loss_cs = self.compute_token_loss(cs_sample, cs_e_out, model, "cs")
            
            en_sample, en_e_out, en_output = self.find_lang(sample, model, "en")
            if en_sample != 0:
                en_loss, en_nll_loss = self.compute_loss(model, en_output, en_sample, reduce=reduce)
                if self.sentence_level:
                    contrastive_loss_en = self.compute_conloss(en_sample, en_e_out, model, "en")
                if self.token_level:
                    token_loss_en = self.compute_token_loss(en_sample, en_e_out, model, "en")

            contrastive_loss = contrastive_loss_fr + contrastive_loss_cs + contrastive_loss_de + contrastive_loss_en
            token_loss = token_loss_de + token_loss_fr + token_loss_cs + token_loss_en

            assert contrastive_loss, "No Contrastive Loss!"

            if de_sample == 0 and en_sample == 0:
                loss = token_loss + contrastive_loss
                nll_loss = loss
            elif de_sample == 0 and en_sample != 0:
                loss = en_loss + token_loss + contrastive_loss
                nll_loss = en_nll_loss
            elif de_sample != 0 and en_sample == 0:
                de_loss, nll_loss = self.compute_loss(model, de_output, de_sample, reduce=reduce)
                loss = de_loss + token_loss + contrastive_loss
            else:
                de_loss, de_nll_loss = self.compute_loss(model, de_output, de_sample, reduce=reduce)
                loss = de_loss + en_loss + token_loss + contrastive_loss
                nll_loss = en_nll_loss + de_nll_loss

        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "contrastive_loss": contrastive_loss,
            "contrastive_loss_de": contrastive_loss_de,
            "contrastive_loss_fr": contrastive_loss_fr,
            "contrastive_loss_cs": contrastive_loss_cs,
            "token_loss": token_loss,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def read_region(self, lang, ids):
        bias = 0
        if lang == 'fr':
            bias = self.de_sen
        elif lang == 'cs':
            bias = self.de_sen + self.fr_sen
        elif lang == 'en':
            bias = self.de_sen + self.fr_sen + self.cs_sen
            lang = 'de'
        first = True
        for i in ids:
            if first:
                image_matrix = np.load(f"/{self.root}/{lang}/{lang}_vit_clip_{i-bias}.npy")
                first = False
            else:
                tmp = np.load(f"/{self.root}/{lang}/{lang}_vit_clip_{i-bias}.npy")
                image_matrix = np.vstack((image_matrix, tmp))

        return torch.tensor(image_matrix).float()


    def find_lang(self, sample, model, lang):
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        ids = sample["id"]
        target = sample["target"]
        prev_output_tokens = sample["net_input"]["prev_output_tokens"]

        first = False
        de_tokens = torch.empty(0, 0)
        de_id = torch.empty(0)
        de_nsentences = 0
        de_target = torch.empty(0, 0)
        de_ntokens = 0
        de_prev_output_tokens = torch.empty(0, 0)
        de_src_length = torch.empty(0)

        num = 0
        dict = model.encoder.dictionary
  
        if lang == "de":
            num = '[DE]' #15
        elif lang == "fr":
            num = '[FR]' #17
        elif lang == "cs":
            num = '[CS]' #14
        elif lang == "en":
            num = '[EN]'

        for i in range(src_tokens.shape[0]):
            if dict.string(src_tokens[i][0].unsqueeze(0)) == num:
                if not first:
                    de_tokens = src_tokens[i, :].unsqueeze(0)
                    de_target = target[i, :].unsqueeze(0)
                    de_id = ids[i].unsqueeze(0)
                    de_prev_output_tokens = prev_output_tokens[i, :].unsqueeze(0)
                    de_src_length = src_lengths[i].unsqueeze(0)
                    de_nsentences += 1
                    first = True
                else:
                    tmp = src_tokens[i, :].unsqueeze(0)
                    de_tokens = torch.cat((de_tokens, tmp), 0)
                    tmp = target[i, :].unsqueeze(0)
                    de_target = torch.cat((de_target, tmp), 0)
                    tmp = ids[i].unsqueeze(0)
                    de_id = torch.cat((de_id, tmp), 0)
                    tmp = src_lengths[i].unsqueeze(0)
                    de_src_length = torch.cat((de_src_length, tmp), 0)
                    tmp = prev_output_tokens[i, :].unsqueeze(0)
                    de_prev_output_tokens = torch.cat((de_prev_output_tokens, tmp), 0)
                    de_nsentences += 1

        de_sample = {
            "id": de_id,
            "nsentences": de_nsentences,
            "net_input": {
                "src_tokens": de_tokens,
                "src_lengths": de_src_length,
                "prev_output_tokens": de_prev_output_tokens,
            },
            "target": de_target,
        }
        if first == False:
            return 0, 0, 0
        de_e_output, de_output = model(**de_sample["net_input"])
        return de_sample, de_e_output, de_output

    def compute_token_loss(self, sample, eout, model, lang):
        ids = sample["id"]
        batch_size = ids.shape[0]
        image_matrix = self.read_region(lang, ids).transpose(0, 1)

        encoder_out = eout.encoder_out
        device = encoder_out.device
        image_matrix = image_matrix.to(device)

        i_attn, _ = model.encoder.selective_attns(encoder_out, image_matrix, image_matrix)
        i_attn = i_attn.transpose(0, 1)
        encoder_out = encoder_out.transpose(0, 1)
        sentence_len = encoder_out.shape[1]
        x = model.encoder.normlayer(i_attn)

        i_norm = F.normalize(x, p=2, dim=-1, eps=1e-5)
        w_norm = F.normalize(encoder_out, p=2, dim=-1, eps=1e-5)
        similarity_matrix = w_norm.matmul(i_norm.transpose(1, 2))

        similarity_matrix = similarity_matrix.to(device)
        sim_exp = torch.exp(similarity_matrix / self.token_tem)
        token_loss = 0
        for i in range(batch_size):
            word_sum = 0
            image_sum = 0
            for j in range(encoder_out.shape[1]):
                word_sum += torch.log(torch.div(sim_exp[i, j, j], sim_exp[i].sum(dim=1)[j]))
                image_sum += torch.log(torch.div(sim_exp[i, j, j], sim_exp[i].sum(dim=0)[j]))
            token_loss += (-word_sum - image_sum)

        return token_loss/sentence_len

    def compute_conloss(self, sample, eout, model, lang):
        ids = sample["id"]
        batch_size = ids.shape[0]

        encoder_out = eout.encoder_out.transpose(0, 1)
        sentence_len = encoder_out.shape[1]
        encoder_padding_mask = (~eout.encoder_padding_mask).float()
        word_feature = (encoder_out * encoder_padding_mask.unsqueeze(-1)).sum(dim=1) / encoder_padding_mask.sum(
            dim=1).unsqueeze(-1)

        device = encoder_out.device
        image_matrix = torch.tensor([0]).to(device)
        wendu = self.sen_tem
        if lang == "de" or lang == "en":
            image_matrix_avg = self.de_image_matrix.to(device)
        elif lang == "fr":
            image_matrix_avg = self.fr_image_matrix.to(device)
        elif lang == "cs":
            image_matrix_avg = self.cs_image_matrix.to(device)

        batch_image_tensor_avg = torch.rand(batch_size, 512).to(device)
        bias = 0
        lamda = 5
        if lang == "de":
            for i in range(batch_size):
                batch_image_tensor_avg[i] = image_matrix_avg[ids[i]]

        elif lang == "fr":
            bias = self.de_sen
            for i in range(batch_size):
                batch_image_tensor_avg[i] = image_matrix_avg[ids[i] - bias]

        elif lang == "cs":
            bias = self.de_sen + self.fr_sen
            for i in range(batch_size):
                batch_image_tensor_avg[i] = image_matrix_avg[ids[i] - bias]
        elif lang == "en":
            bias = self.de_sen + self.fr_sen + self.cs_sen
            for i in range(batch_size):
                batch_image_tensor_avg[i] = image_matrix_avg[ids[i] - bias]

        i_norm = F.normalize(batch_image_tensor_avg, p=2, dim=-1, eps=1e-5)
        w_norm = F.normalize(word_feature, p=2, dim=-1, eps=1e-5)
        similarity_matrix = w_norm.matmul(i_norm.transpose(0, 1))
        similarity_matrix = similarity_matrix.to(device)
        sim_exp = torch.exp(similarity_matrix / wendu)

        word_sum = 0
        image_sum = 0

        for i in range(batch_size):
            word_sum += torch.log(torch.div(sim_exp[i, i], sim_exp.sum(dim=1)[i]))
            image_sum += torch.log(torch.div(sim_exp[i, i], sim_exp.sum(dim=0)[i]))
        contrastive_loss = lamda * (-word_sum - image_sum)

        return contrastive_loss


    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        con_loss_sum = sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        con_loss_sum_de = sum(log.get("contrastive_loss_de", 0) for log in logging_outputs)
        con_loss_sum_fr = sum(log.get("contrastive_loss_fr", 0) for log in logging_outputs)
        con_loss_sum_cs = sum(log.get("contrastive_loss_cs", 0) for log in logging_outputs)
        token_loss_sum = sum(log.get("token_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "con_loss", con_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "con_loss_de", con_loss_sum_de / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "con_loss_fr", con_loss_sum_fr / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "con_loss_cs", con_loss_sum_cs / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "token_loss", token_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
