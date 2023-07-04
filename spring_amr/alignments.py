import re, time
import numpy as np
import networkx as nx
from collections import defaultdict
import penman
import torch
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

class WAG(nx.DiGraph):
    def rand_contract(self, prob=0.0, start_node=0, end_node=0, relabel=True):
        '''
        Contracting random nodes with their neighbours. Each neighbour is sampled randomly too
        :param prob: probability
        :param start_node: node from which we start to sample
        :return:
        '''
        samples_nodes = np.arange(start_node, end_node)
        mask = np.random.binomial(size=samples_nodes.size, n=1, p=prob) == 1

        for node in samples_nodes[mask]:
            self = self.contract_with_random_neighbour(node)

        if relabel:
            kept_nodes = samples_nodes[mask == False]
            mapping = dict(zip(list(kept_nodes), list(range(start_node, start_node + len(kept_nodes)))))
            self = nx.relabel_nodes(self, mapping)

        return self, mask == False

    def contract_with_random_neighbour(self, node):
        nbrs = list(set(self.neighbors(node)) - {node})
        if nbrs:
            neighbor = np.random.choice(nbrs)
            return nx.contracted_nodes(self, neighbor, node, self_loops=False)
        else:
            self.remove_node(node)

        return self

def process_xml_tags(text):
    repl_pattern = '#${id}$#'
    matches = set(re.findall(r'<+[a-zA-Z/][^>]*?>+', text))
    tags_dict = dict()
    for i, m in enumerate(matches):
        str_id = repl_pattern.format(id=str(i))
        text = text.replace(m, str_id)
        tags_dict[str_id] = m

    return text, tags_dict

def triples_to_graph(tokenizer, graph, with_relations=True, self_loops=True, undirected=False,
                     keep_full=False, align_from_metadata=False, connect_sub_tokens_to_preds=False):
    ''' ...
    Converts triples v1, rel, v2 to WAG
    '''
    triples = graph.align_epidata
    metadata_tok, tags_dict = process_xml_tags(graph.metadata['tok'])
    words = [w if w not in tags_dict else tags_dict[w] for w in metadata_tok.split()]

    G = WAG()
    nodes = defaultdict(set)
    parents = defaultdict(set) # used for contraction
    child_counter = defaultdict(int)
    node_hash = {list(triples.items())[0][0][0]: '1'}
    rel_counter = 0

    for triple, alignment in triples.items():
        v = {}
        v[0], rel, v[1] = triple
        for i in range(2):
            if v[i][0] == '"' and v[i][-1] == '"':
                v[i] = v[i].replace('"', '')
        v1, v2 = v[0], v[1]
        # Assigning to relations unique names
        rel = rel + '_' + str(rel_counter)
        rel_counter += 1

        if rel[:9] == ':instance':
            for a in alignment:
                if isinstance(a, penman.surface.Alignment):
                    # Filtering is for silver alignmenets bc they have sometimes wrong indices
                    nodes[v1] |= set(filter(lambda x: x < len(words), a.indices))
                    G.add_node(v1)
            if align_from_metadata:
                G.add_node(v1)
        else:
            for a in alignment:
                if isinstance(a, penman.surface.Alignment):
                    # Filtering is for silver alignmenets bc they have sometimes wrong indices
                    nodes[v2] |= set(filter(lambda x: x < len(words), a.indices))
                elif with_relations and isinstance(a, penman.surface.RoleAlignment):
                    nodes[rel] |= set(filter(lambda x: x < len(words), a.indices))
            if with_relations:
                G.add_edge(v1, rel)
                G.add_edge(rel, v2)
                parents[rel].add(v1)
                parents[v2].add(rel)
            else:
                G.add_edge(v1, v2)
                parents[v2].add(v1)

            if align_from_metadata:
                __v1, __v2 = (v1, v2) if v1 in node_hash else (v2, v1)
                child_counter[__v1] += 1
                node_hash[__v2] = node_hash[__v1] + '.' + str(child_counter[__v1])
                node_hash[rel] = node_hash[__v2] + '.r'

    if align_from_metadata:
        new_labels = defaultdict(set)
        align_dict = get_align_dict(graph.metadata['alignments'])
        hash_to_node = {h: n for n, h in node_hash.items()}
        for hash, token_index in align_dict.items():
            if hash in hash_to_node:
                new_labels[hash_to_node[hash]].add(token_index)

        nodes = new_labels

    def contract_with_child(G, node_name):
        cur_edges = list(G.edges(node_name))
        if len(cur_edges):
            closest_node = cur_edges[0][1]
            G = nx.contracted_nodes(G, closest_node, node_name, self_loops=False)
            return G, False
        else:
            return G, True

    if not keep_full:
        remove_nodes = []
        # Make it reverse since originally it goes from top to bottom
        for node_name in reversed(list(G.nodes)):
            if not len(nodes[node_name]):
                # No alignment for node, thus we do contraction for it to closest parent
                if len(parents[node_name]):
                    # Parent with the highest degree
                    parent_node = \
                    sorted(list(parents[node_name]), key=lambda n: G.degree[n] if G.has_node(n) else -1, reverse=True)[
                        0]
                    if G.has_node(parent_node):
                        G = nx.contracted_nodes(G, parent_node, node_name, self_loops=False)
                    else:
                        G, del_flag = contract_with_child(G, node_name)
                        if del_flag:
                            remove_nodes.append(node_name)
                else:
                    G, del_flag = contract_with_child(G, node_name)
                    if del_flag:
                        remove_nodes.append(node_name)

        for node_name in remove_nodes:
            G.remove_node(node_name)

    # Constructing dictionary of indices for corresponding names
    idx_to_name = defaultdict(list)
    for k, v in nodes.items():
        for idx in list(v):
            idx_to_name[idx].append(k)
    contracted = dict()
    for idx, names in idx_to_name.items():
        main_node = names[0]
        if main_node in contracted:
            main_node = contracted[main_node]
        if len(names) > 1:
            # Contracting nodes if one word corresponds to multiple nodes
            for i in range(1, len(names)):
                if G.has_node(names[i]):
                    G = nx.contracted_nodes(G, main_node, names[i], self_loops=False)
                    contracted[names[i]] = main_node
        idx_to_name[idx] = main_node
    # Make nodes again since we contracted nodes
    nodes = defaultdict(set)
    for idx, name in idx_to_name.items():
        nodes[name].add(idx)

    shift = 1
    all_tokens = [tokenizer.bos_token_id]
    old_to_new_index = dict()

    for i, word in enumerate(words):
        tokens = tokenizer.encode(word)[1:-1]
        all_tokens += tokens
        if i in idx_to_name:
            mod_node_index = i + shift
            old_to_new_index[i] = mod_node_index
            node_name = idx_to_name[i]
            if G.has_node(node_name):
                G = nx.relabel_nodes(G, {node_name: mod_node_index})
                pred_nodes = G.pred[mod_node_index]
                # Adding new nodes of tokens (word-parts) connecting them to the first token
                for k in range(len(tokens) - 1):
                    sub_token_node_index = mod_node_index + k + 1
                    G.add_edge(sub_token_node_index, mod_node_index)
                    # Connecting parents to sub-tokens
                    if connect_sub_tokens_to_preds:
                        for _pred_node in pred_nodes:
                            G.add_edge(_pred_node, sub_token_node_index)

                shift += len(tokens) - 1
            else:
                old_index = list(nodes[node_name] & set(old_to_new_index.keys()) - {i})[0]
                c_edges = G.edges(old_to_new_index[old_index])
                if len(c_edges):
                    for from_node, to_node in c_edges:
                        if from_node != to_node:
                            G.add_edge(mod_node_index, to_node)
                else:
                    G.add_node(mod_node_index)
    all_tokens += [tokenizer.eos_token_id]
    # Make undirected
    if undirected:
        for v1, v2 in G.edges:
            G.add_edge(v2, v1)
    # Adding self-loops
    if self_loops:
        for i in range(len(all_tokens)):
            G.add_edge(i, i)

    if keep_full:
        node_names = {v1: v2 for (v1, r, v2), alignment in triples.items() if r == ':instance'}
        G = nx.relabel_nodes(G, node_names)

    return G, all_tokens


def get_align_dict(alignments):
    '''
    The function is used to construct dictionary with alignments coming from ::alignments field
    '''
    align_dict = dict()
    for alignment in alignments.split():
        dash_index = alignment.find('-')
        token_index, graph_path = int(alignment[:dash_index]), alignment[dash_index + 1:]
        align_dict[graph_path] = token_index

    return align_dict


def prepare_full_graph(tokenizer, G, last_index, self_loops=False, nodes_limit=5000, mask_probability=0.10):
    '''
    1) Replacing node labels with numbers (node ids).
    2) Masking non-aligned (extra) nodes
    '''
    extra_nodes = []
    remove_nodes = []
    vertex_index = last_index

    for v in G.nodes:
        if type(v) == str:
            vertex_index += 1
            if vertex_index <= last_index + nodes_limit:
                if np.random.uniform() <= mask_probability:
                    # Add mask token instead of real tokens
                    extra_nodes.append((v, vertex_index, [tokenizer.mask_token_id]))
                else:
                    extra_nodes.append((v, vertex_index, node_name_token_ids(tokenizer, v)))
            else:
                remove_nodes.append(v)

    new_labels = {old: new for old, new, _ in extra_nodes}
    G = nx.relabel_nodes(G, new_labels)
    G.remove_nodes_from(remove_nodes)

    token_ids = [torch.tensor(t) for _, _, t in extra_nodes]
    if len(token_ids):
        token_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    else:
        token_ids = torch.tensor([])

    if self_loops:
        for _, node, _ in extra_nodes:
            G.add_edge(node, node)

    return G, token_ids


def node_name_token_ids(tokenizer, node_name, rem_sufix=r'\_\d+$'):
    if rem_sufix:
        node_name = re.sub(rem_sufix, '', node_name)

    token_name = tokenizer.INIT + node_name
    if token_name in tokenizer.added_tokens_encoder:
        return [tokenizer.added_tokens_encoder[token_name]]
    else:
        return tokenizer.encode(node_name)[1:-1]


def plot_graph(nx_g, labels=None, remove_self_loops=True):
    if remove_self_loops:
        nodes_before = set([e[0] for e in nx_g.edges] + [e[1] for e in nx_g.edges])
        nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
        nodes_after = set([e[0] for e in nx_g.edges] + [e[1] for e in nx_g.edges])
        nx_g.remove_nodes_from(nodes_before - nodes_after)

    plt.figure(figsize=(12, 12))
    pos = graphviz_layout(nx_g, prog="dot")

    nx.draw(nx_g, with_labels=True, pos=pos, labels=labels,
            node_color='white')

    plt.axis('off')
    plt.show()

def lables_for_plot(tokenizer, G, all_tokens):
    str_all_tokens = {i: tok_id for i, tok_id in enumerate(tokenizer.convert_ids_to_tokens(all_tokens))}
    labels = {}
    for n in G.nodes:
        if n in str_all_tokens:
            labels[n] = str_all_tokens[n] #[1:]
        else:
            labels[n] = re.sub('_\d+', '', n)

    return labels

