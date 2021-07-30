import streamlit as st
from annotated_text import annotated_text
import nltk
import pandas as pd
import os, sys
sys.path.append('./entity_detection/nn/')
sys.path.append('./relation_prediction/nn/')
sys.path.append('./evidence_integration')

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import torch
import numpy as np

device_name = 'cuda'
device_id = '0'
device = torch.device('{}:{}'.format(device_name, device_id))

project_directory = os.path.abspath('.')
data_directory = os.path.abspath('.')

@st.cache
def load_model(model_path):
  model_entity = torch.load(model_path, map_location=lambda storage, location: storage.cuda(device))
  return model_entity

model_entity_path = os.path.join(data_directory, 'entity_detection/nn/saved_checkpoints/lstm/id1_best_model.pt')
model_entity = load_model(model_entity_path)

st.title('Strong Baselines for Simple Question Answering over Knowledge Graphs with and without Neural Networks')

st.markdown('This demonstration is based on the following [paper](https://aclanthology.org/N18-2047/).')
user_input_question = st.text_input("Simple question", "What is Jeff Hall known for?")
user_input_question_tokenized = " ".join(nltk.word_tokenize(user_input_question))

## Entity detection
st.markdown('## Entity detection')
st.markdown('Given a question, the goal of entity detection is to identify the entity being queried. '
        'This is naturally formulated as a sequence labeling problem, where for each token, '
        'the task is to assign one of two tags, either `ENTITY` or `NOTENTITY`.')

from sq_entity_dataset import SQdataset
from torchtext.legacy import data

@st.cache
def load_simple_questions(dataset_path):
  TEXT = data.Field(lower=True)
  ED = data.Field()

  train, dev, test = SQdataset.splits(TEXT, ED, path=dataset_path)
  TEXT.build_vocab(train, dev, test)
  ED.build_vocab(train, dev, test)
  return TEXT, ED

class Sample:
  def __init__(self, text):
    self.text = text

TEXT, ED = load_simple_questions(os.path.join(data_directory, 'data/processed_simplequestions_dataset'))

user_input_question_tokenized_preprocess = TEXT.preprocess(user_input_question_tokenized)
user_input_question_pt = TEXT.process([user_input_question_tokenized_preprocess]).to(device)
scores = model_entity(Sample(user_input_question_pt))
scores = torch.max(scores, 1)[1]

user_input_question_with_annotated_entities = []
for s, t in zip(scores.cpu().detach().numpy().tolist(), user_input_question_tokenized.split(' ')):
  if s == 3:
    user_input_question_with_annotated_entities.append((t+" ", "ENT", "#faa"))
  else:
    user_input_question_with_annotated_entities.append(t+" ")

st.markdown('We detected the following entity in the input question:')
annotated_text(*user_input_question_with_annotated_entities)

predicted_entity = ' '.join([t for (s, t) in zip(scores, user_input_question_tokenized_preprocess) if s == 3])

## Entity linking
st.markdown('## Entity linking')
st.markdown('The output of entity detection is a sequence of tokens representing a candidate entity. '
            'This still needs to be linked to an actual node in the knowledge graph. '
            'In Freebase, each node is denoted by a Machine Identifier, or MID. The method uses fuzzy string matching.')

import pickle
from collections import defaultdict
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz

@st.cache
def load_index_n_grams(file):
  with open(file, "rb") as handler:
      global  inverted_index
      inverted_index = pickle.load(handler)
      inverted_index = defaultdict(str, inverted_index)
  return inverted_index

def get_ngram(text):
  # ngram = set()
  ngram = []
  tokens = text.split()
  for i in range(len(tokens) + 1):
    for j in range(i):
      if i - j <= 3:
        # ngram.add(" ".join(tokens[j:i]))
        temp = " ".join(tokens[j:i])
        if temp not in ngram:
          ngram.append(temp)
  # ngram = list(ngram)
  ngram = sorted(ngram, key=lambda x: len(x.split()), reverse=True)
  return ngram

freebase_n_grams_index_file = os.path.join(data_directory, 'indexes/entity_2M.pkl')
inverted_index = load_index_n_grams(freebase_n_grams_index_file)

tokens = get_ngram(predicted_entity)
st.markdown("We are looking for the following entity in Freebase \"**{}**\".".format(predicted_entity))
st.markdown("We consider all the following n-grams: **{}**.".format(tokens))
st.markdown('Once all candidate entities have been gathered, they are then ranked by Levenshtein Distance '
            'to the MID’s canonical label.')

stopword = set(stopwords.words('english'))
C = []

if len(tokens) > 0:
    maxlen = len(tokens[0].split())
for item in tokens:
    if len(item.split()) < maxlen and len(C) == 0:
        maxlen = len(item.split())
    if len(item.split()) < maxlen and len(C) > 0:
        break
    if item in stopword:
        continue
    print(item, inverted_index[item])
    C.extend(inverted_index[item])

HITS_TOP_ENTITIES = 100
C_scored = []

for mid_text_type in sorted(set(C)):
    score_entity = fuzz.ratio(mid_text_type[1], predicted_entity) / 100.0
    # C_counts format : ((mid, text, type), score_based_on_fuzz)
    C_scored.append((mid_text_type, score_entity))

C_scored.sort(key=lambda t: t[1], reverse=True)
cand_mids = C_scored[:HITS_TOP_ENTITIES]
cand_mids = [(cm[0][0], cm[0][1], cm[0][2], cm[1]) for cm in cand_mids]
df = pd.DataFrame(cand_mids, columns=['Freebase identifier', 'Freebase label', 'Freebase type', 'Score'])

st.markdown("We retrieved all the following entities:")

st.dataframe(df)

## Relation prediction
st.markdown('## Relation prediction')
st.markdown('The goal of relation prediction is to identify the relation being queried. '
            'This step is achieved as classification over the entire question.')


from sq_relation_dataset import SQdataset as SQdatasetRelations

@st.cache
def load_simple_questions_relations(dataset_path):
  TEXT = data.Field(lower=True)
  RELATION = data.Field(sequential=False)
  train, dev, test = SQdatasetRelations.splits(TEXT, RELATION,path=dataset_path)
  TEXT.build_vocab(train, dev, test)
  RELATION.build_vocab(train, dev)
  return TEXT, RELATION

TEXT, RELATION = load_simple_questions_relations(os.path.join(data_directory, 'data/processed_simplequestions_dataset'))
index2tag = np.array(RELATION.vocab.itos)

model_relation_path = os.path.join(data_directory, 'relation_prediction/nn/saved_checkpoints/cnn/id1_best_model.pt')
model_relation = load_model(model_relation_path)

scores = model_relation(Sample(user_input_question_pt))
top_k_scores, top_k_indices = torch.topk(scores, k=5, dim=1, sorted=True)

top_k_relatons_array = [index2tag[tag] for tag in top_k_indices[0].cpu().detach().numpy().tolist()]
rels = list(zip(top_k_relatons_array, top_k_scores[0].cpu().detach().numpy().tolist()))
rels_df = pd.DataFrame(rels, columns=['Freebase relation identifier', 'Freebase relation label'])

st.dataframe(rels_df)

## Evidence integration
st.markdown('## Evidence integration')
st.markdown('Given the top `m` entities and `r` relations from the previous components, '
            'the final task is to integrate evidence to arrive at a single (entity, relation) prediction.')

import math
from util import clean_uri, processed_text, www2fb, rdf2fb

@st.cache
def load_index(filename):
  with open(filename, 'rb') as handler:
    index = pickle.load(handler)
  return index

@st.cache
def get_mid2wiki(filename):
  mid2wiki = defaultdict(bool)
  fin = open(filename)
  for line in fin.readlines():
    items = line.strip().split('\t')
    sub = rdf2fb(clean_uri(items[0]))
    mid2wiki[sub] = True
  return mid2wiki

mid2wiki = get_mid2wiki(os.path.join(data_directory, 'data/fb2w.nt'))
index_reach = load_index(os.path.join(data_directory, 'indexes/reachability_2M.pkl'))
index_degrees = load_index(os.path.join(data_directory, 'indexes/degrees_2M.pkl'))

id2answers = []

for (mid, mid_name, mid_type, mid_score) in cand_mids:
  for (rel, rel_log_score) in rels:
    # if this (mid, rel) exists in FB
    if rel in index_reach[mid]:
      rel_score = math.exp(float(rel_log_score))
      comb_score = (float(mid_score)**0.6) * (rel_score**0.1)
      id2answers.append((mid, rel, mid_name, mid_type, mid_score, rel_score, comb_score, int(mid2wiki[mid]), int(index_degrees[mid][0])))

id2answers.sort(key=lambda t: (t[6], t[3],  t[7], t[8]), reverse=True)

id2answers_df = pd.DataFrame(id2answers, columns=['Freebase entity identifier', 'Freebase relation label', 'Entity',
                                                  'Frebase entity label', 'Entity score', 'Relation score',
                                                  'Global score', 'Present in Wikipedia', 'Wikipedia score'])
st.dataframe(id2answers_df)

st.markdown('Please run the following command to query the freebase knowledge base:')

with st.echo():
  st.write('zgrep \"<http://rdf.freebase.com/ns/{}>   <http://rdf.freebase.com/ns/{}>\" -m 5 freebase-rdf-latest.gz'.format(id2answers[0][1][3:], id2answers[0][3][3:]))
