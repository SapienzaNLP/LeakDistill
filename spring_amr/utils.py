
from glob import glob
from pathlib import Path
import re
from collections import defaultdict
import networkx as nx
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from transformers import AutoConfig

from spring_amr.dataset import AMRDataset, AMRDatasetTokenBatcherAndLoader
from spring_amr.modeling_bart import AMRBartForConditionalGeneration
from spring_amr.tokenizer_bart import AMRBartTokenizer, PENMANBartTokenizer
#from spring_amr.modeling_mbart import AMRMBartForConditionalGeneration

from transformers.models.bart.tokenization_bart_fast import *


def instantiate_model_and_tokenizer(
        name='facebook/bart-large',
        checkpoint=None,
        additional_tokens_smart_init=True,
        dropout=0.15,
        attention_dropout=0.15,
        from_pretrained=True,
        init_reverse=False,
        collapse_name_ops=False,
        penman_linearization=False,
        use_pointer_tokens=False,
        raw_graph=False,
        language="en_XX",
        mode="amr",
        direction = "amr",
        adapter_configs=None,
        output_hidden_states=False
):
    if raw_graph:
        assert penman_linearization

    skip_relations = False

    tokenizer_name = name

    config = AutoConfig.from_pretrained(name)
    config.output_past = False
    config.no_repeat_ngram_size = 0
    config.prefix = " "
    config.output_attentions = True
    config.dropout = dropout
    config.attention_dropout = attention_dropout
    config.adapter_configs = adapter_configs
    config.output_hidden_states = output_hidden_states
 
    tokenizer_type = None
    snt_tokenizer_type = None
    model_type = None

    if penman_linearization and mode == "amr" and language == "en_XX":
        snt_tokenizer_type = BartTokenizerFast
        tokenizer_type = PENMANBartTokenizer
        model_type = AMRBartForConditionalGeneration
    else:
        raise NotImplementedError
    src_lang = language
    tgt_lang = "en_XX"

    if penman_linearization:
        tokenizer = tokenizer_type.from_pretrained(
            tokenizer_name,
            collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens,
            raw_graph=raw_graph,
            config=config,
            direction=direction,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            add_prefix_space=True,
        )
    else:
        tokenizer = tokenizer_type.from_pretrained(
            tokenizer_name,
            collapse_name_ops=collapse_name_ops,
            use_pointer_tokens=use_pointer_tokens,
            config=config,
            direction=direction,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            add_prefix_space=True,
        )

    snt_tokenizer = snt_tokenizer_type.from_pretrained(
        tokenizer_name,
        collapse_name_ops=collapse_name_ops,
        use_pointer_tokens=use_pointer_tokens,
        config=config,
        add_prefix_space=True,
    )
    model = model_type.from_pretrained(name, config=config) if from_pretrained else model_type(config)
    model.resize_token_embeddings(len(tokenizer))

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu')['model'], strict=False)
    else:
        if additional_tokens_smart_init:
            modified = 0
            for tok in tokenizer.added_tokens_list:
                idx = tokenizer.convert_tokens_to_ids(tok)

                tok = tok.lstrip(tokenizer.INIT)

                if idx < tokenizer.vocab_size:
                    continue

                elif tok.startswith('<pointer:') and tok.endswith('>'):
                    tok_split = ['pointer', str(tok.split(':')[1].strip('>'))]

                elif tok.startswith('<'):
                    continue

                elif tok.startswith(':'):

                    if skip_relations:
                        continue

                    elif tok.startswith(':op'):
                        tok_split = ['relation', 'operator', str(int(tok[3:]))]

                    elif tok.startswith(':snt'):
                        tok_split = ['relation', 'sentence', str(int(tok[4:]))]

                    elif tok.startswith(':ARG'):
                        tok_split = ['relation', 'argument', str(int(tok[4:]))]

                    elif mode == "amr":
                        tok_split = ['relation'] + tok.lstrip(':').split('-')

                    else:
                        tok_split = ['relation'] + tok.lstrip(':').split('_')

                else:
                    tok_split = tok.split('-')

                tok_split_ = tok_split
                tok_split = []
                for s in tok_split_:
                    s_ = s + tokenizer.INIT
                    if (tokenizer.unk_token != s_ and tokenizer.convert_tokens_to_ids(s_) != tokenizer.unk_token_id):
                        tok_split.append(s_)
                    else:
                        tok_split.extend(tokenizer._tok_bpe(s))

                vecs = []
                for s in tok_split:
                    idx_split = tokenizer.convert_tokens_to_ids(s)
                    if idx_split != tokenizer.unk_token_id:
                        vec_split = model.model.shared.weight.data[idx_split].clone()
                        vecs.append(vec_split)

                if vecs:
                    vec = torch.stack(vecs, 0).mean(0)
                    noise = torch.empty_like(vec)
                    noise.uniform_(-0.1, +0.1)
                    model.model.shared.weight.data[idx] = vec + noise
                    modified += 1

        model.model.set_input_embeddings(model.model.shared)
        if init_reverse:
            model.init_reverse_model()

    return model, tokenizer, snt_tokenizer


def instantiate_loader(
        glob_pattn,
        tokenizer,
        snt_tokenizer,
        batch_size=500,
        evaluation=True,
        out=None,
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        raw_data=True,
        alignment_paths=None,
        random_reduce=None,
        align_keep_full=True,
        align_from_metadata=False,
        mlm_prob_upper_bound=0,
        extra_nodes_mask_prob=0,
        extra_nodes_contract_prob=0,
        snt_to_tok=False
):
    paths = []
    if isinstance(glob_pattn, str) or isinstance(glob_pattn, Path):
        glob_pattn = [glob_pattn]
    for gpattn in glob_pattn:
        paths += [Path(p) for p in glob(gpattn)]
    if evaluation and alignment_paths is None:
        assert out is not None
        Path(out).write_text(
            '\n\n'.join([p.read_text() for p in paths]))

    align_paths = None
    if alignment_paths is not None:
        align_paths = []
        glob_pattn = alignment_paths
        if isinstance(glob_pattn, str) or isinstance(glob_pattn, Path):
            glob_pattn = [glob_pattn]
        for gpattn in glob_pattn:
            align_paths += [Path(p) for p in glob(gpattn)]
        if evaluation:
            Path(out).write_text(
                '\n\n'.join([Path(str(p).replace('alignments', 'amrs')).read_text() for p in align_paths]))


    dataset = AMRDataset(
        paths,
        tokenizer,
        snt_tokenizer,
        use_recategorization=use_recategorization,
        remove_longer_than=remove_longer_than,
        remove_wiki=remove_wiki,
        dereify=dereify,
        raw_data=raw_data,
        alignment_paths=align_paths,
        reduce_prob=random_reduce,
        align_keep_full=align_keep_full,
        align_from_metadata=align_from_metadata,
        mlm_prob_upper_bound=mlm_prob_upper_bound,
        extra_nodes_mask_prob=extra_nodes_mask_prob,
        extra_nodes_contract_prob=extra_nodes_contract_prob,
        snt_to_tok=snt_to_tok
    )

    loader = AMRDatasetTokenBatcherAndLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not evaluation
    )
    return loader

def instantiate_loader_graphs(
        graphs,
        tokenizer,
        snt_tokenizer,
        batch_size=500,
        evaluation=True,
        out=None,
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
):

    dataset = AMRDataset(
        graphs,
        tokenizer,
        snt_tokenizer,
        use_recategorization=use_recategorization,
        remove_longer_than=remove_longer_than,
        remove_wiki=remove_wiki,
        dereify=dereify,
        raw_data=False
    )
    loader = AMRDatasetTokenBatcherAndLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not evaluation,
    )
    return loader

def class_to_dict(cls):
    d = vars(cls)
    return {k: d[k] for k in d if k[:2] != '__'}

def dict_to_class(d):
    class O:
        pass
    for k in d:
        setattr(O, k, d[k])
    return O

