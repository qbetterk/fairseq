# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask, load_langpair_dataset
from fairseq.utils import new_arange


NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise", "full_mask"])

@dataclass
class TranslationLevenshteinConfig(TranslationConfig):
    noise: NOISE_CHOICES = field(
        default="random_mask",
        metadata={
            "help": "type of noise"
        },
    )

@register_task("sar_dst", dataclass=TranslationLevenshteinConfig)
class DSTSemiAutoRegressiveTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    cfg: TranslationLevenshteinConfig

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=False,
        )

    def init_token_sar(self, target_tokens):
        """
        target_old : <s>  d1  t1 v1 , d2 t2 v2 ,  ...  </s> <pad> ...
        target_new :   d1  t1 v1 ,  </s>   d2  t2 v2 ,  ...  </s> <pad> ...
        prev_target: <unk> d1 t1 v1  ,   <unk> d2 t2 v2 ...   ,   <pad>
        """
        pad = self.tgt_dict.pad()
        bos = self.tgt_dict.bos()
        eos = self.tgt_dict.eos()
        unk = self.tgt_dict.unk()

        target = target_tokens.clone()
        # pad mask
        pad_mask = target.eq(pad)
        # replace <eos> with <unk>
        target[target==eos] = unk
        # right shift every token
        unk_col = target.new_full((target.size(0), 1), unk)
        # add initiall token <unk> and remove the last
        target = torch.cat([unk_col, target[:,:-1]], dim=-1).long()
        # mask with <pad>
        target = target.masked_fill_(pad_mask, pad)

        return target

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
        from fairseq.sar_generator import SARGenerator

        return SARGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", False),
            retain_history=getattr(args, "retain_iter_history", False),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        sample["prev_target"] = self.init_token_sar(sample["target"])
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample["prev_target"] = self.init_token_sar(sample["target"])
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
