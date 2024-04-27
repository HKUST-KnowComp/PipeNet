import torch
import networkx as nx
import itertools
import json
from tqdm import tqdm
from utils.conceptnet import merged_relations
import numpy as np
from scipy import sparse
import pickle
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool
from functools import partial
from collections import OrderedDict
import os

from utils.maths import *

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def concepts2adj(node_ids):
    global id2relation
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1

    # cids += 1  # note!!! index 0 is reserved for padding
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids


def concepts_to_adj_matrices_2hop_all_pair(data, prune_rate=0.9):
    qc_ids, ac_ids, qa_nodes, dic = data
    extra_nodes = set()

    all_concepts = dic['all_concepts']
    all_map_id = dic['all_map_id']
    spans_sort = dic['spans_sort']
    spans_dist = dic['spans_dist']

    all_concepts_map = {}
    # concept --> span list index
    for concept, map_id in zip(all_concepts, all_map_id):
        all_concepts_map[concept] = map_id

    # span id from grounding, varied query to query
    # grouned one-hop concepts, propagate dist score from span to concept

    qa_map_parse_id = []
    for qa_concept in dic['all_concepts']:
        if len(all_concepts_map) == 0:
            print(qa_concept)
            print(all_concepts_map)
            continue
        qa_map_parse_id.append(all_concepts_map[qa_concept])

    qa_concept_dist = np.zeros((len(dic['all_concepts']), len(dic['all_concepts'])))
    if not len(all_concepts_map) == 0:
        for i in range(len(dic['all_concepts'])):
            for j in range(len(dic['all_concepts'])):
                qa_concept_dist[i][j] = spans_dist[qa_map_parse_id[i]][qa_map_parse_id[j]]

    # concept : score
    # one-hop matched concept dist score
    schema_score_dic = {}
    # find for ac
    target_index = []
    
    for k in range(len(dic['ac'])):
        if dic['ac'][k] in dic['all_concepts']:
            ac_index = dic['all_concepts'].index(dic['ac'][k])
            target_index.append(ac_index)

    for k in range(len(dic['all_concepts'])):
        if len(target_index) > 0:
            score_k = np.array([qa_concept_dist[k, j] for j in target_index])
        else:
            score_k = qa_concept_dist[k,:]
        # {concept: score}
        schema_score_dic[dic['all_concepts'][k]] = -np.mean(score_k)

    for ac in dic['ac']:
        # For some nodes do not shown in qa, should be very few (hard grounding)
        if ac not in schema_score_dic:
            schema_score_dic[ac] = -0.0

    # two-hop matched concept dist score, qa_nodes are in the form of conceptnet id
    extra_nodes_h_t = {}
    for hid in qa_nodes:
        for tid in qa_nodes:
            if hid != tid and hid in cpnet_simple.nodes and tid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[hid]) & set(cpnet_simple[tid])
                for extra_node_ in set(cpnet_simple[hid]) & set(cpnet_simple[tid]):
                    #{id: neighbor1_concept, neighbor2_concept}
                    extra_nodes_h_t[extra_node_] = (id2concept[hid], id2concept[tid])

    q_only_ids = qa_nodes - ac_ids
    extra_nodes = extra_nodes - q_only_ids - ac_ids

    # id : score
    schema_ids = list(sorted(q_only_ids)) + list(sorted(ac_ids)) + list(sorted(extra_nodes))
    schema_score_list = []
    for id_ in sorted(q_only_ids):
        score_ = schema_score_dic[id2concept[id_]]
        schema_score_list.append(score_)

    for id_ in sorted(ac_ids):
        score_ = schema_score_dic[id2concept[id_]]
        schema_score_list.append(score_)

    for extra_node_ in sorted(extra_nodes):
        h, t = extra_nodes_h_t[extra_node_]
        h_score = schema_score_dic[h]
        t_score = schema_score_dic[t]
        schema_score_list.append(np.mean([h_score, t_score]))

    cid2score = OrderedDict(sorted(list(zip(schema_ids, schema_score_list)), key=lambda x: -x[1])) # from high to low
    extra_nodes_new = sorted(extra_nodes, key=lambda x: -cid2score[x]) # from high to low

    # prune extra node here
    schema_graph = sorted(q_only_ids) + sorted(ac_ids) + extra_nodes_new[:int(len(extra_nodes_new)*(1.0-prune_rate))]
    arange = np.arange(len(schema_graph))
    qmask = arange < len(q_only_ids)
    amask = (arange >= len(q_only_ids)) & (arange < (len(q_only_ids) + len(ac_ids)))
    schema_graph_score = [cid2score[key] for key in schema_graph]

    adj, concepts = concepts2adj(schema_graph)
    # no need of cid2score
    return adj, concepts, qmask, amask, None


def generate_adj_data_from_grounded_concepts(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes, prune=0.9):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')
    print(f'pruning rate: ', prune)

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    qa_data = []
    with open(grounded_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            a_ids = set(concept2id[c] for c in dic['ac'])
            all_ids = set(concept2id[c] for c in dic['all_concepts'])
            qa_data.append((q_ids, a_ids, all_ids, dic))

    concepts_to_adj_matrices_2hop_all_pair_p = partial(concepts_to_adj_matrices_2hop_all_pair, prune_rate=prune)
    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair_p, qa_data), total=len(qa_data)))

    output_dir = '/'.join(output_path.split('/')[:-1])
    if not os.path.exists(output_dir):
        os.makefirs(output_dir)

    # res is a list of tuples, each tuple consists of four elements (adj, concepts, qmask, amask)
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()


if __name__ == '__main__':

    split = 'train'
    generate_adj_data_from_grounded_concepts('../data/csqa/grounded/{}.grounded.new.jsonl'.format(split),
                                                  '../data/cpnet/conceptnet.en.pruned.graph',
                                                  '../data/cpnet/concept.txt',
                                                  '../data/csqa/graph/{}.graph.new.adj.pk'.format(split), 1)


