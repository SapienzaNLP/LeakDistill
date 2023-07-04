import logging
import random, copy
import torch
import numpy as np
from cached_property import cached_property
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask
from spring_amr.IO import read_raw_amr_data, read_amr_data
import penman
from spring_amr.alignments import *
import spring_amr.snt_to_tok as snttotok


def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['lm_labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:, :-1]
    lm_labels = x['input_ids'][:, 1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'lm_labels': lm_labels}
    return x, y


class AMRDataset(Dataset):

    def __init__(
            self,
            paths,
            tokenizer,
            snt_tokenizer,
            device=torch.device('cpu'),
            use_recategorization=False,
            remove_longer_than=None,
            remove_wiki=False,
            dereify=True,
            alignment_paths=None,
            raw_data=True,
            reduce_prob=None,
            align_keep_full=True,
            full_graph_extra_nodes_limit=500,
            align_from_metadata=False,
            add_seq_edges=True, # Add sequential edges from tokens order
            mlm_prob_upper_bound=0,
            replace_bos_with_leak_token=False,
            extra_nodes_mask_prob=0,
            extra_nodes_contract_prob=0,
            snt_to_tok=False
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.snt_tokenizer = snt_tokenizer
        self.device = device
        self.with_alignment = alignment_paths is not None
        self.align_keep_full = align_keep_full
        self.full_graph_extra_nodes_limit = full_graph_extra_nodes_limit
        self.align_from_metadata = align_from_metadata
        self.add_seq_edges = add_seq_edges
        self.mlm_prob_upper_bound = mlm_prob_upper_bound
        self.replace_bos_with_leak_token = replace_bos_with_leak_token
        self.extra_nodes_mask_prob = extra_nodes_mask_prob
        self.extra_nodes_contract_prob = extra_nodes_contract_prob

        if self.with_alignment:
            graphs = read_raw_amr_data(alignment_paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
            for g in graphs:
                if 'tok' not in g.metadata:
                    g.metadata['tok'] = g.metadata['snt']
                if '@' in g.metadata['tok']:
                    g.metadata['tok'] = re.sub(r'@([^@]+?)@', '\\1', g.metadata['tok'])
                g.metadata['snt'] = g.metadata['tok']
        else:
            if raw_data:
                graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
            else:
                graphs = read_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)

            if snt_to_tok:
                # Making the same input as it was trained
                for g in graphs:
                    g.metadata['snt'] = snttotok.tokenize(g.metadata['snt'])

        self.graphs = []
        self.sentences = []
        self.ids = []
        self.linearized = []
        self.linearized_extra = []
        self.tok_snts = []
        self.align_graphs = {}
        self.align_contracted_graphs = {}
        self.all_tokens_ids = {}
        self.all_extra_tokens_ids = {}
        self.k_step_matrices = {}
        self.remove_longer_than = remove_longer_than

        for i, g in enumerate(graphs):
            if reduce_prob is not None and random.random() >= reduce_prob:
                continue
            l, e = self.tokenizer.linearize(self.remove_align_from_graph(g))

            try:
                tok_snt, _ = self.batch_encode_sentences([g.metadata['snt']])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            self.tok_snts.append(tok_snt['input_ids'][0])
            self.sentences.append(g.metadata['snt'])
            self.ids.append(g.metadata['id'])
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        sample['sentences_id'] = self.ids[idx]
        sample['tok_sentences'] = self.tok_snts[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])
        if 'masked_tok' in self.graphs[idx].metadata:
            sample['masked_tok'] = self.graphs[idx].metadata['masked_tok']

        return sample

    def size(self, sample):
        return len(sample['linearized_graphs_ids'])

    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        x, extra = self.batch_encode_sentences(x, device=device)

        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        extra['sentences_id'] = [s['sentences_id'] for s in samples]
        if self.with_alignment:
            edges, contracted_edges, neg_edges, batch_k_step_matrices, batch_tokens_ids, (batch_og_tokens, batch_og_masks) = self.collate_allign_data(extra['ids'],
                                                                                                           device)
            extra['align_graph_edges'] = edges
            extra['align_graph_neg_edges'] = neg_edges
            extra['align_contracted_edges'] = contracted_edges
            extra['k_step_matrices'] = batch_k_step_matrices
            x['input_ids'] = batch_tokens_ids
            x['attention_mask'] = (batch_tokens_ids != self.tokenizer.pad_token_id).int()
            # Extra extra
            extra['orig_graph_data'] = batch_og_tokens, batch_og_masks

            extra['leak_input_ids'] = x['input_ids']
            if self.replace_bos_with_leak_token:
                first_column = torch.tensor([[self.tokenizer.leak_token_id] * x['input_ids'].shape[0]]).T.to(x['input_ids'].device)
                extra['leak_input_ids'] = torch.cat((first_column, x['input_ids'][:, 1:]), dim=1)

            if 'masked_tok' in samples[0]:
                x_aux = [s['masked_tok'] for s in samples]
                x_aux, _ = self.batch_encode_sentences(x_aux, device=device)
                extra['masked_sentences_x'] = x_aux

        return x, y, extra

    def batch_encode_sentences(self, sentences, device=torch.device('cpu')):
        sentences = [s for s in sentences]
        extra = {'sentences': sentences}
        if self.tokenizer.direction == "text":
            batch = self.snt_tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True)
            batch["input_ids"][batch["input_ids"] == self.convert_tokens_to_ids(self.src_lang)] = 1
            batch["input_ids"] = torch.roll(batch["input_ids"], 1, 1)
            batch["input_ids"][:, 0] = self.convert_tokens_to_ids(self.src_lang)
        else:
            batch = self.snt_tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True)

        batch = {k: v.to(device) for k, v in batch.items()}
        return batch, extra

    def remove_align_from_graph(self, g):
        g.align_epidata = dict(g.epidata)
        for k, arr in g.epidata.items():
            g.epidata[k] = [a for a in arr if not (
                        isinstance(a, penman.surface.Alignment) or isinstance(a, penman.surface.RoleAlignment))]
        return g

    def get_align_data(self, idx):
        extra_tokens = []
        if idx not in self.align_graphs:
            # Normal case
            if self.align_keep_full:
                G, tokens_ids = triples_to_graph(self.snt_tokenizer, self.graphs[idx],
                                                 self_loops=True, keep_full=True, align_from_metadata=self.align_from_metadata)
                G, extra_tokens = prepare_full_graph(self.tokenizer, G, len(tokens_ids) - 1,  self_loops=True,
                                                 nodes_limit=self.full_graph_extra_nodes_limit,
                                                 mask_probability=self.extra_nodes_mask_prob)
            else:
                G, tokens_ids = triples_to_graph(self.snt_tokenizer, self.graphs[idx], self_loops=True,
                                                 keep_full=False, align_from_metadata=self.align_from_metadata)
            contracted_G = G
            self.align_graphs[idx] = G
            self.align_contracted_graphs[idx] = contracted_G
            self.all_tokens_ids[idx] = tokens_ids
            self.all_extra_tokens_ids[idx] = extra_tokens
            self.k_step_matrices[idx] = torch.tensor([])

        return self.align_graphs[idx], self.align_contracted_graphs[idx], self.all_tokens_ids[idx], self.all_extra_tokens_ids[idx], self.k_step_matrices[idx]

    def mask_tokens(self, tokens):
        '''
        Masking with probability sampled from uniform(0, mlm_prob_upper_bound)
        '''
        device = tokens.device
        prob = float(torch.FloatTensor(1).uniform_(0, self.mlm_prob_upper_bound))
        mask_len = tokens.shape[0]
        probs = torch.tensor([0] + [prob] * (mask_len - 2) + [0])
        tokens[torch.bernoulli(probs).to(device).bool()] = self.snt_tokenizer.mask_token_id
        return tokens

    def collate_allign_data(self, ids, device):
        edges = []
        num_nodes_list = []
        neg_edges = []
        contracted_edges = []
        batch_tokens_ids = []
        batch_extra_tokens = []
        full_graph_masks = None
        batch_k_step_matrices = []

        for idx in ids:
            G, contracted_G, tokens_ids, extra_tokens, k_step_matrices = self.get_align_data(idx)
            if self.extra_nodes_contract_prob > 0:
                G = copy.deepcopy(G)
                G, kept_mask = G.rand_contract(prob=self.extra_nodes_contract_prob,
                                               start_node=len(tokens_ids),
                                               end_node=len(tokens_ids) + extra_tokens.shape[0])
                extra_tokens = extra_tokens[kept_mask]

            num_nodes_list.append(len(G.nodes))
            edges.append(self.get_edge_tensor(G, device, len(tokens_ids), add_seq_order_edge=self.add_seq_edges))
            contracted_edges.append(torch.tensor(list(contracted_G.edges)).t().to(device))

            batch_tokens_ids.append(self.mask_tokens(torch.tensor(tokens_ids).to('cpu')))
            batch_k_step_matrices.append(k_step_matrices.to(device))
            # Extra
            if self.align_keep_full:
                batch_extra_tokens.append(extra_tokens.to(device))

        if self.align_keep_full:
            full_graph_masks = self.full_graph_masks([t.to(device) for t in batch_tokens_ids], batch_extra_tokens)
            batch_extra_tokens, extra_tokens_len_mask = self.pad_extra_nodes(batch_extra_tokens)
            full_graph_masks += (extra_tokens_len_mask,)

        edges = self.edge_batch_to_single_graph(edges, num_nodes_list)
        contracted_edges = self.edge_batch_to_single_graph(contracted_edges, num_nodes_list)

        batch_tokens_ids = torch.nn.utils.rnn.pad_sequence(batch_tokens_ids, batch_first=True,
                                                           padding_value=self.snt_tokenizer.pad_token_id).to(device)

        return edges, contracted_edges, neg_edges, batch_k_step_matrices, batch_tokens_ids, (batch_extra_tokens, full_graph_masks)

    def full_graph_masks(self, batch_token_ids, batch_extra_tokens):
        '''
        Produce masks for original, extra and pad tokens in order to access them in BART encoder.
        These masks are used in case extra_nodes_as_input=True to set hidden states for non-aligned nodes
        :param batch_token_ids:
        :param batch_extra_tokens:
        :return: masks of positions of original, extra and pad tokens
        '''
        device = batch_token_ids[0].device
        padding_value = self.tokenizer.pad_token_id
        extra_token_mask_id = -10
        batch_token_info = []

        for i, token_ids in enumerate(batch_token_ids):
            batch_token_info.append(torch.cat(
                (token_ids,
                torch.full((batch_extra_tokens[i].shape[0],), fill_value=extra_token_mask_id, device=device))
            ))

        all_tokens = torch.nn.utils.rnn.pad_sequence(batch_token_info, batch_first=True, padding_value=padding_value)
        extra_tokens_mask = (all_tokens == extra_token_mask_id)
        padding_mask = (all_tokens == padding_value)
        tokens_mask = torch.logical_not(torch.logical_or(extra_tokens_mask, padding_mask))
        return tokens_mask, extra_tokens_mask, (padding_mask == False)

    def pad_extra_nodes(self, batch_extra_tokens):
        # Fixing empty tensor
        batch_extra_tokens = [et if et.shape[0] else torch.tensor([[1]]).to(et.device) for et in batch_extra_tokens]
        # Calculating shared size of a batched tensor
        max_height = max(et.shape[0] for et in batch_extra_tokens)
        max_width = max(et.shape[1] for et in batch_extra_tokens)
        # Padding and stacking tensors
        batch_list = [torch.nn.functional.pad(et, (0, max_width - et.shape[1], 0, max_height - et.shape[0]), value=1) for et in
                                batch_extra_tokens]

        batch_extra_tokens = torch.stack(batch_list)

        length_mask = (batch_extra_tokens[:, :, :1] != 1).squeeze(-1)
        return batch_extra_tokens, length_mask

    def edge_batch_to_single_graph(self, edges, num_nodes_list):
        shift = np.cumsum([0] + num_nodes_list)
        edges = [e + shift[i] for i, e in enumerate(edges)]
        return torch.cat(edges, dim=1)

    def get_edge_tensor(self, G, device, seq_len=None, add_seq_order_edge=False):
        edge_list = list(G.edges)
        if add_seq_order_edge and seq_len is not None:
            edge_list += self.sequential_edges(seq_len)
        return torch.tensor(edge_list).t().to(device)

    def sequential_edges(self, seq_len, as_torch_geom=False):
        edges = list(zip(range(1, seq_len - 2), range(2, seq_len - 1)))
        if as_torch_geom:
            edges = torch.tensor(edges).t()

        return edges


class AMRDatasetTokenBatcherAndLoader:
    def __init__(self, dataset, batch_size=800, device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]

        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()
