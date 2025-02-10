# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
import os
import time
from functools import wraps
from typing import Dict, Optional, Tuple

import numpy
import torch

from megatron.training import get_args
from megatron.core.datasets.utils import Split, log_single_rank
from megatron.core.datasets.gpt_dataset import (_build_document_index,
                                                _build_shuffle_index
                                                )
from mindspeed_llm.tasks.utils.error_utils import GPTDatasetSampleIndexError
from .blended_megatron_dataset_builder import need_to_build_dataset

logger = logging.getLogger(__name__)


def gpt_dataset_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # Adapt to MTP
        _args = get_args()
        self.num_nextn_predict_layers = _args.num_nextn_predict_layers
        fn(self, *args, **kwargs)

    return wrapper


def gpt_dataset_getitem_func(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
    """Abstract method implementation

    Args:
        idx (Optioal[int]): The index into the dataset

    Returns:
        Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
    """
    if idx is None:
        # Batch padding sequence so the index does not matter
        text, _ = self._query_document_sample_shuffle_indices(0)
    else:
        text, _ = self._query_document_sample_shuffle_indices(idx)

    text = torch.from_numpy(text).long()
    if self.config.add_extra_token_to_sequence:
        tokens = text[:-1].contiguous()
        labels = text[1:].contiguous()
    else:
        tokens = text
        labels = torch.roll(text, shifts=-1, dims=0)
        labels[-1] = self._pad_token_id

    if (
        not self.masks_and_position_ids_are_cacheable
        or not self.masks_and_position_ids_are_cached
    ):
        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            tokens,
            self.config.tokenizer.eod,
            self.config.reset_position_ids,
            self.config.reset_attention_mask,
            self.config.eod_mask_loss,
            self.config.create_attention_mask,
        )
        if self.masks_and_position_ids_are_cacheable:
            self.cached_attention_mask = attention_mask
            self.cached_loss_mask = loss_mask
            self.cached_position_ids = position_ids
            self.masks_and_position_ids_are_cached = True
    else:
        attention_mask = self.cached_attention_mask
        loss_mask = self.cached_loss_mask
        position_ids = self.cached_position_ids

    # For padded sequences, mask the loss
    # Adapt to MTP
    loss_mask[labels[:labels.shape[0] - self.num_nextn_predict_layers] == self._pad_token_id] = 0.0

    # For padded sequences, ensure the embedding layer can map the token ID
    tokens[tokens == self._pad_token_id] = 0
    labels[labels == self._pad_token_id] = 0

    # Batch padding sequence so we mask the loss
    if idx is None:
        loss_mask = torch.zeros_like(loss_mask)

    if self.config.create_attention_mask:
        return {
            "tokens": tokens,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
    else:
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }


def _query_document_sample_shuffle_indices(
    self, idx: int
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Get the text (token ids) and document ids for a given index

    Args:
        idx (int): The index into the dataset

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
    """
    # Do the shuffle mapping
    idx = self.shuffle_index[idx]

    # Get the beginning and end documents and offsets
    doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
    doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

    document_ids = []
    sample_parts = []

    # Sample spans a single document
    if doc_index_beg == doc_index_end:
        # Add the document id
        document_ids.append(self.document_index[doc_index_beg])

        # Add the entire sample
        # Adapt to MTP
        sample_parts.append(
            self.dataset.get(
                self.document_index[doc_index_beg],
                offset=doc_index_beg_offset,
                length=doc_index_end_offset
                - doc_index_beg_offset
                + self.config.add_extra_token_to_sequence + self.num_nextn_predict_layers,
            )
        )

    # Sample spans multiple documents
    else:
        for i in range(doc_index_beg, doc_index_end + 1):
            # Add the document id
            document_ids.append(self.document_index[i])

            # Add the sample part
            offset = 0 if i > doc_index_beg else doc_index_beg_offset
            # Adapt to MTP
            length = (
                None
                if i < doc_index_end
                else doc_index_end_offset + self.config.add_extra_token_to_sequence + self.num_nextn_predict_layers
            )
            sample_parts.append(
                self.dataset.get(self.document_index[i], offset=offset, length=length)
            )
    assert len(document_ids) == len(
        sample_parts
    ), f"len(document_ids) ({len(document_ids)}) != len(sample_parts) ({len(sample_parts)})"

    length = sum(map(len, sample_parts))

    # Pad the sample if necessary
    # Adapt to MTP
    if length < (
            self.config.sequence_length + self.config.add_extra_token_to_sequence + self.num_nextn_predict_layers):
        sample_parts.append(
            [self._pad_token_id]
            * (
                        self.config.sequence_length + self.config.add_extra_token_to_sequence +
                        self.num_nextn_predict_layers - length)
        )

    return (
        numpy.concatenate(sample_parts, dtype=numpy.int64),
        numpy.array(document_ids, dtype=numpy.int64),
    )


def _build_document_sample_shuffle_indices(
    self,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Build the document index, the sample index, and the shuffle index

    The document index:
        -- 1-D
        -- An ordered array of document ids

    The sample index:
        -- 2-D
        -- The document indices and offsets which mark the start of every sample

    The shuffle index:
        -- 1-D
        -- A random permutation of index range of the sample index

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The document index, the sample index, and the shuffle index
    """
    path_to_cache = self.config.path_to_cache
    if path_to_cache is None and not self.config.mock:
        path_to_cache = os.path.join(
            self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
        )

    # start of megatron_adaptation,
    # here we change from (class)GPTDataset._build_document_sample_shuffle_indices
    # end of megatron_adaptation

    if path_to_cache:
        get_path_to = lambda suffix: os.path.join(
            path_to_cache, f"{self.unique_description_hash}-{type(self).__name__}-{suffix}"
        )
        path_to_description = get_path_to("description.txt")
        path_to_document_index = get_path_to("document_index.npy")
        path_to_sample_index = get_path_to("sample_index.npy")
        path_to_shuffle_index = get_path_to("shuffle_index.npy")
        cache_hit = all(
            map(
                os.path.isfile,
                [
                    path_to_description,
                    path_to_document_index,
                    path_to_sample_index,
                    path_to_shuffle_index,
                ],
            )
        )
    else:
        cache_hit = False

    if not path_to_cache or (
            not cache_hit
            and (not torch.distributed.is_initialized() or need_to_build_dataset())
    ):

        log_single_rank(
            logger,
            logging.INFO,
            f"Build and save the {type(self).__name__} {self.index_split.name} indices",
        )
        self.built_anew_on_cache_miss = True
        t_beg = time.time()


        sequence_length = self.config.sequence_length
        num_tokens_per_epoch = self._get_num_tokens_per_epoch()
        num_epochs = self._get_num_epochs(num_tokens_per_epoch)

        if num_epochs == 1:
            separate_final_epoch = False
        else:
            # Get the number of samples for the last epoch
            # Adapt to MTP
            num_samples_sans_final_epoch = (
                                                   (num_epochs - 1) * num_tokens_per_epoch
                                                   - self.config.add_extra_token_to_sequence
                                                   - self.num_nextn_predict_layers
                                           ) // sequence_length
            num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
            num_samples_per_epoch = (
                                            num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                                            - self.num_nextn_predict_layers
                                    ) // sequence_length

            # num_samples_from_final_epoch should be non-negative
            assert num_samples_from_final_epoch >= 0

            # num_samples_from_final_epoch should not exceed max value
            assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

            # Separate the final epoch if it falls below the threshold
            threshold = 0.80
            separate_final_epoch = num_samples_from_final_epoch < int(
                threshold * num_samples_per_epoch
            )

            log_single_rank(
                logger,
                logging.DEBUG,
                f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
            )
            log_single_rank(logger, logging.DEBUG, f"> threshold: {threshold}")
            log_single_rank(
                logger, logging.DEBUG, f"> num_samples_per_epoch: {num_samples_per_epoch}"
            )

        log_single_rank(
            logger, logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}"
        )

        numpy_random_state = numpy.random.RandomState(self.config.random_seed)

        # Build the document index
        document_index = _build_document_index(
            self.indices, num_epochs, numpy_random_state, separate_final_epoch
        )
        drop_last_partial_sequence = True
        if self.index_split == Split.valid:
            drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence

        # Build the sample index
        from megatron.core.datasets import helpers

        if self.index_split == Split.valid:
            drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence
        else:
            drop_last_partial_sequence = True

        assert document_index.dtype == numpy.int32
        assert self.dataset.sequence_lengths.dtype == numpy.int32
        if len(document_index) * 2 > len(self.dataset.sequence_lengths):
            # Heuristic: if "access density" of sequence_lengths is relatively high,
            # force loading the mmap-ed array into memory by taking a copy.
            # System performance benefits come from two aspects:
            # 1. **sequentially** pre-loading the whole file if we're gonna read a large fraction anyways.
            # 2. GIL is held when calling into c++ code; making the c++ func faster improves parallelism.
            sequence_lengths_for_cpp = self.dataset.sequence_lengths.copy()
        else:
            sequence_lengths_for_cpp = self.dataset.sequence_lengths
        # Adapt to MTP
        sample_index = helpers.build_sample_idx(
            sequence_lengths_for_cpp,
            document_index,
            sequence_length,
            num_epochs,
            num_tokens_per_epoch,
            drop_last_partial_sequence,
            self.config.add_extra_token_to_sequence + self.num_nextn_predict_layers,
        )

        if any(sample_index[:, 0] < 0):
            _url = "https://gitee.com/ascend/MindSpeed-LLM/wikis/megatron%20data%20helpers%E5%8F%AF%E8%83%BD%E5%BC%95%E5%85%A5%E7%9A%84%E9%97%AE%E9%A2%98"
            raise GPTDatasetSampleIndexError(f"Bad sample index. Visit {_url} for more information")

        # Build the shuffle index
        if separate_final_epoch:
            shuffle_index = _build_shuffle_index(
                num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
            )
        else:
            shuffle_index = _build_shuffle_index(
                sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
            )

        if path_to_cache:
            os.makedirs(path_to_cache, exist_ok=True)
            # Write the description
            with open(path_to_description, "wt") as writer:
                writer.write(self.unique_description)
            numpy.save(path_to_document_index, document_index, allow_pickle=True)
            numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
            numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
        else:
            log_single_rank(
                logger,
                logging.WARNING,
                f"Unable to save the {type(self).__name__} indexes because path_to_cache is None",
            )
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
        )
        log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

        return document_index, sample_index, shuffle_index

    log_single_rank(
        logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"
    )

    log_single_rank(
        logger,
        logging.INFO,
        f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
    )
    t_beg = time.time()
    document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode='r')
    t_end = time.time()
    log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

    log_single_rank(
        logger,
        logging.INFO,
        f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
    )
    t_beg = time.time()
    sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode='r')

    if any(sample_index[:, 0] < 0):
        _url = "https://gitee.com/ascend/MindSpeed-LLM/wikis/megatron%20data%20helpers%E5%8F%AF%E8%83%BD%E5%BC%95%E5%85%A5%E7%9A%84%E9%97%AE%E9%A2%98"
        raise GPTDatasetSampleIndexError(f"Bad sample index. Visit {_url} for more information")

    t_end = time.time()
    log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

    log_single_rank(
        logger,
        logging.INFO,
        f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
    )
    t_beg = time.time()
    shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode='r')
    t_end = time.time()
    log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

    log_single_rank(
        logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
    )

    return document_index, sample_index, shuffle_index


def _get_num_tokens_per_epoch(self) -> int:
    """Calculate the number of tokens in a single epoch

    Returns:
        int: The number of tokens in a single epoch
    """
    return int(numpy.sum(self.dataset.sequence_lengths[self.indices])) + self.num_nextn_predict_layers


def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
    """Calculate the number of epochs

    Args:
        num_tokens_per_epoch (int): The number of tokens in a single epoch

    Returns:
        int: The number of epochs
    """
    num_epochs = 1
    num_tokens = num_tokens_per_epoch
    if self.num_samples is None:
        return num_epochs
    else:
        # Adapt to MTP
        num_tokens_requested = (
            self.num_samples * self.config.sequence_length
        ) + self.config.add_extra_token_to_sequence + self.num_nextn_predict_layers
        while num_tokens < num_tokens_requested:
            num_epochs += 1
            num_tokens += num_tokens_per_epoch
    return num_epochs


def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
    create_attention_mask: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

        create_attention_mask (bool): Switch to enable the attention masks generation. Can be disabled if attention kernel generates masks by itself.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention

        torch.Tensor: The mask used for loss value during training

        torch.Tensor: The position ID's of the token
    """
    args = get_args()
    seq_length = data.numel() - args.num_nextn_predict_layers

    if create_attention_mask:
        attention_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=data.device)
        ).unsqueeze(0)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids
