from multiprocessing import Pool
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import nltk
import json
import string
import numpy as np
import networkx as nx
import stanza
nlp2 = stanza.Pipeline('en', tokenize_no_ssplit=True)


__all__ = ['create_matcher_patterns', 'ground']


# the lemma of it/them/mine/.. is -PRON-

blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])


nltk.download('stopwords', quiet=True)
nltk_stopwords = nltk.corpus.stopwords.words('english')

# CHUNK_SIZE = 1

CPNET_VOCAB = None
PATTERN_PATH = None
nlp = None
matcher = None


def load_cpnet_vocab(cpnet_vocab_path):
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    return cpnet_vocab


def create_pattern(nlp, doc, debug=False):
    pronoun_list = set(["my", "you", "it", "its", "your", "i", "he", "she", "his", "her", "they", "them", "their", "our", "we"])
    # Filtering concepts consisting of all stop words and longer than four words.
    if len(doc) >= 5 or doc[0].text in pronoun_list or doc[-1].text in pronoun_list or \
            all([(token.text in nltk_stopwords or token.lemma_ in nltk_stopwords or token.lemma_ in blacklist) for token in doc]):
        if debug:
            return False, doc.text
        return None  # ignore this concept as pattern

    pattern = []
    for token in doc:  # a doc is a concept
        pattern.append({"LEMMA": token.lemma_})
    if debug:
        return True, doc.text
    return pattern


def create_matcher_patterns(cpnet_vocab_path, output_path, debug=False):
    cpnet_vocab = load_cpnet_vocab(cpnet_vocab_path)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    docs = nlp.pipe(cpnet_vocab)
    all_patterns = {}

    if debug:
        f = open("filtered_concept.txt", "w")

    for doc in tqdm(docs, total=len(cpnet_vocab)):

        pattern = create_pattern(nlp, doc, debug)
        if debug:
            if not pattern[0]:
                f.write(pattern[1] + '\n')

        if pattern is None:
            continue
        all_patterns["_".join(doc.text.split(" "))] = pattern

    print("Created " + str(len(all_patterns)) + " patterns.")
    with open(output_path, "w", encoding="utf8") as fout:
        json.dump(all_patterns, fout)
    if debug:
        f.close()


def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_", " "))
    lcs = set()
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs


def load_matcher(nlp, pattern_path):
    with open(pattern_path, "r", encoding="utf8") as fin:
        all_patterns = json.load(fin)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in all_patterns.items():
        matcher.add(concept, None, pattern)
    return matcher


def ground_qa_pair(qa_pair):
    global nlp, matcher
    if nlp is None or matcher is None:
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
        matcher = load_matcher(nlp, PATTERN_PATH)

    s, a = qa_pair
    all_concepts, all_concepts_list, all_concepts_map_id, span_sort, span_dists =  ground_mentioned_concepts(nlp, nlp2, matcher, s, find=True)
    answer_concepts, _, _, _, _ = ground_mentioned_concepts(nlp, nlp2, matcher, a)
    question_concepts = all_concepts - answer_concepts

    if len(question_concepts) == 0:
        question_concepts = hard_ground(nlp, s, CPNET_VOCAB)  # not very possible

    if len(answer_concepts) == 0:
        answer_concepts = hard_ground(nlp, a, CPNET_VOCAB)  # some case

    question_concepts = sorted(list(question_concepts))
    answer_concepts = sorted(list(answer_concepts))
    return {"sent": s, "ans": a, "qc": question_concepts, 'ac': answer_concepts, "all_concepts": all_concepts_list, "all_map_id": all_concepts_map_id, \
            "spans_sort": span_sort, "spans_dist": span_dists}


def ground_mentioned_concepts(nlp, nlp2, matcher, s, find=False):

    #s = "Sammy wanted to go to where the people were.  race track might he go."
    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    if find == True:
        doc2 = nlp2(s)

        # more than one sentence
        doc2_dep = doc2.sentences[0].to_dict() 
        G = nx.Graph()
        nodes = [parsed_txt['id'] for parsed_txt in doc2_dep]
        edges = [(parsed_txt['id'], parsed_txt['head']) for parsed_txt in doc2_dep]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        p = nx.shortest_path(G)

    mentioned_concepts = set()
    span_to_concepts = {}
    span_to_concepts_indexs = {}

    # token_id: (start, end)
    for match_id, start, end in matches:
        span = doc[start:end].text  # the matched span
        # keep consistent with nlp2
        span_to_concepts_indexs[span] = (start+1, end+1)

        original_concept = nlp.vocab.strings[match_id]

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        # one span may match several concepts
        span_to_concepts[span].add(original_concept)

    # find the shortest dists between spans
    span_dists = []
    span_sort = []
    if find == True:
        for span, inds in span_to_concepts_indexs.items():
            span_sort.append(span)
            span_dist_ = []
            start, end = inds
            for span2, inds2 in span_to_concepts_indexs.items():
                if span == span2:
                    span_dist_.append(0.0)
                    continue
                start2, end2 = inds2
                dist = []
                for i in range(start, end):
                    for j in range(start2, end2):
                        try:
                            path_len = len(p[i][j]) - 1.0 #edge len = num of nodes - 1
                        except:
                            path_len = 100.0
                        dist.append(path_len)
                # choose the min distance of words between spans
                span_dist_.append(np.min(dist))
            span_dists.append(span_dist_)

    mentioned_concepts_to_span = {}
    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        concepts_sorted.sort(key=len)

        # each span use at most three matched concepts
        shortest = concepts_sorted[0:3]

        if find == True:
            # index all the concepts
            for c in concepts_sorted:
                mentioned_concepts_to_span[c] = span_sort.index(span)

        for c in shortest:
            if c in blacklist:
                continue

            # a set with one string like: set("like_apples")
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect) > 0:
                mentioned_concepts.add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)

        # if a mention exactly matches with a concept
        # search over all the concepts
        exact_match = set([concept for concept in concepts_sorted if concept.replace("_", " ").lower() == span.lower()])
        assert len(exact_match) < 2
        mentioned_concepts.update(exact_match)

    # match mentioned concepts with span
    mentioned_concepts_list = list(mentioned_concepts)
    if find == True:
        mentioned_concepts_map_id = [mentioned_concepts_to_span[mention_concept] for mention_concept in mentioned_concepts_list]
    else:
        mentioned_concepts_map_id = None

    return mentioned_concepts, mentioned_concepts_list, mentioned_concepts_map_id, span_sort, span_dists

def hard_ground(nlp, sent, cpnet_vocab):
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
    sent = " ".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    try:
        assert len(res) > 0
    except Exception:
        print(f"for {sent}, concept not found in hard grounding.")
    return res


def match_mentioned_concepts(sents, answers, num_processes):
    res = []
    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(ground_qa_pair, zip(sents, answers)), total=len(sents)))
    return res


# To-do: examine prune
def prune(data, cpnet_vocab_path):
    # reload cpnet_vocab
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]

    prune_data = []
    for item in tqdm(data):
        qc = item["qc"]
        prune_qc = []
        for c in qc:
            if c[-2:] == "er" and c[:-2] in qc:
                continue
            if c[-1:] == "e" and c[:-1] in qc:
                continue
            have_stop = False
            # remove all concepts having stopwords, including hard-grounded ones
            for t in c.split("_"):
                if t in nltk_stopwords:
                    have_stop = True
            if not have_stop and c in cpnet_vocab:
                prune_qc.append(c)

        ac = item["ac"]
        prune_ac = []
        for c in ac:
            if c[-2:] == "er" and c[:-2] in ac:
                continue
            if c[-1:] == "e" and c[:-1] in ac:
                continue
            all_stop = True
            for t in c.split("_"):
                if t not in nltk_stopwords:
                    all_stop = False
            if not all_stop and c in cpnet_vocab:
                prune_ac.append(c)

        try:
            assert len(prune_ac) > 0 and len(prune_qc) > 0
        except Exception as e:
            pass

        item["qc"] = prune_qc
        item["ac"] = prune_ac

        prune_data.append(item)
    return prune_data


def ground(statement_path, cpnet_vocab_path, pattern_path, output_path, num_processes=10, debug=False):
    global PATTERN_PATH, CPNET_VOCAB
    if PATTERN_PATH is None:
        PATTERN_PATH = pattern_path
        CPNET_VOCAB = load_cpnet_vocab(cpnet_vocab_path)

    sents = []
    answers = []
    with open(statement_path, 'r') as fin:
        lines = [line for line in fin]

    if debug:
        lines = lines[192:195]
        print(len(lines))
    for line in lines:
        if line == "":
            continue
        j = json.loads(line)

        for statement in j["statements"]:
            # already get answer concepts in statement
            sents.append(statement["statement"])

        for answer in j["question"]["choices"]:
            ans = answer['text']
            try:
                assert all([i != "_" for i in ans])
            except Exception:
                print(ans)
            answers.append(ans)

    #res = match_mentioned_concepts(sents, answers, num_processes)
    #res = prune(res, cpnet_vocab_path)

    res = []
    for sent, answer in zip(sents, answers):
        ret_ = ground_qa_pair((sent, answer))
        #prune qc and ac
        ret = prune([ret_], cpnet_vocab_path)
        res.extend(ret)

    # check_path(output_path)
    with open(output_path, 'w') as fout:
        for dic in res:
            fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()


if __name__ == "__main__":
    #create_matcher_patterns("../data/cpnet/concept.txt", "./matcher_res.txt", True)
    #split = 'train'
    split = 'test'
    #split = 'test'
    ground('./data/csqa/statement/{}.statement.jsonl'.format(split),
           './data/cpnet/concept.txt',
           './data/cpnet/matcher_patterns.json',
           './data/csqa/grounded/{}.grounded.jsonl'.format(split))
