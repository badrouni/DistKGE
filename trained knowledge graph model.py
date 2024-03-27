# -*- coding: utf-8 -*-



# Required Installations and Imports
"""

# may take around 5-10 minutes
!pip install ogb
!python -c "import torch; print(torch.__version__)"
!python -c "import torch; print(torch.version.cuda)"
!pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-geometric
!pip install -q git+https://github.com/snap-stanford/deepsnap.git

import numpy as np
import ogb
import os
import pdb
import random
import torch
import torch_geometric
import tqdm
from ogb.linkproppred import LinkPropPredDataset, PygLinkPropPredDataset
from torch.utils.data import DataLoader, Dataset

"""# Dataset Preparation

We focus here on a collaboratively collected knowledge base known by the name of **FB15K-237**, which is derived from Freebase repository. FB15K-237 is a cleaner version of its original counterpart FB15k with inverse relations removed and consists of 14541 entities and 237 relations.
"""



meta_dict = {'eval metric': 'mrr',
             'task type': 'KG_completion',
             'download_name': 'fb15k-237',
             'version': '1',
             'url': 'https://drive.google.com/file/d/1NoebmUcqHheuWCyye6e3zm_rsbM6gvyj/view?usp=sharing',
             'add_inverse_edge': False,
             'has_node_attr': False,
             'has_edge_attr': False,
             'split': 'msr',
             'additional node files': 'None',
             'additional edge files': 'edge_reltype',
             'is hetero': False,
             'binary': False,
             'dir_path': './fb15k-237-v1',
             }
fb_dataset = PygLinkPropPredDataset(name='ogb-fb15k237', meta_dict=meta_dict)

# split the dataset using the ogb function
split_edge = fb_dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]

true_edges = torch.load('./fb15k-237-v1/processed/data_processed')

train_edge

"""# Relation Dataset

We define our dataset class here that generates both positive and negative  triples for training.
"""

class RelationDataset(Dataset):
  def __init__(self, edges, true_edges, filter=False):
    self.true_edges = true_edges
    self.train_edges = edges
    self.edge_index = edges['edge_index']
    self.edge_reltype = edges['edge_reltype']
    self.num_nodes = edges['num_nodes']
    self.num_rels = 237
    self.rel_dict = {}
    self.true_edge_dict = {}
    self.filter = filter

    # We construct a dictionary that maps edges to relation types
    # We do this to quickly filter out postive edges while sampling negative
    # edges.
    for i in range(self.true_edges['edge_index'].shape[1]):
      h = self.true_edges['edge_index'][0, i]
      t = self.true_edges['edge_index'][1, i]
      r = self.true_edges['edge_reltype'][i, 0]
      if (h,t) not in self.true_edge_dict:
        self.true_edge_dict[(h,t)] = []
      self.true_edge_dict[(h,t)].append(r)

  def __len__(self):
    return self.edge_index.size(1)

  def _sample_negative_edge(self, idx):
    sample = random.uniform(0, 1)
    found = False
    while not found:
      if sample <= 0.4:
        # corrupt the head entity
        h = self.edge_index[0, idx]
        t = torch.randint(0, self.num_nodes, (1,))
        r = self.edge_reltype[idx,:]
      elif 0.4 < sample < 0.8:
        # corrupt the tail entity
        t = self.edge_index[1, idx]
        h = torch.randint(0, self.num_nodes, (1,))
        r = self.edge_reltype[idx,:]
      else:
        # corrupt the relation
        # adding this auxilliary loss is shown to improve performance
        t = self.edge_index[1, idx]
        h = self.edge_index[0, idx]
        r = torch.randint(0, self.num_rels, (1,))
      if not self.filter:
        found = True
      else:
        # check if the edge is a true edge
        if (h, t) not in self.true_edge_dict:
          found = True
        elif r not in self.true_edge_dict[(h, t)]:
          found = True
    data = [torch.tensor([h,t]), r]
    return data

  def __getitem__(self, idx):
    pos_sample = [self.edge_index[:, idx], self.edge_reltype[idx,:]]
    neg_sample = self._sample_negative_edge(idx)
    return pos_sample, neg_sample

class TestRelationDataset(Dataset):
  def __init__(self, edges, true_edges, filter=False, num_neg=14000, mode='head'):
    self.true_edges = true_edges
    self.edge_index = edges['edge_index']
    self.edge_reltype = edges['edge_reltype']
    self.num_nodes = edges['num_nodes']
    self.num_neg = num_neg
    self.mode = mode
    self.true_edge_dict = {}
    self.filter = filter

    # We construct a dictionary that maps edges to relation types
    # We do this to quickly filter out postive edges while sampling negative
    # edges.
    for i in range(self.true_edges['edge_index'].shape[1]):
      h = self.true_edges['edge_index'][0, i]
      t = self.true_edges['edge_index'][1, i]
      r = self.true_edges['edge_reltype'][i, 0]
      if (h,t) not in self.true_edge_dict:
        self.true_edge_dict[(h,t)] = []
      self.true_edge_dict[(h,t)].append(r)

  def __len__(self):
    return self.edge_index.size(1)

  def _sample_negative_edge(self, idx, mode):
    triples = []
    random_node_idx = list(range(self.num_nodes))
    random.shuffle(random_node_idx)
    for n in random_node_idx:
      r = self.edge_reltype[idx,:]
      if mode == 'head':
        # corrupt tail if in head mode
        t = torch.tensor(n)
        h = self.edge_index[0, idx]
      elif mode == 'tail':
        # corrupt head if in tail mode
        h = torch.tensor(n)
        t = self.edge_index[1, idx]
      ht = torch.tensor([h, t])
      if self.filter:
        # check if edge is present in the knowledge graph
        if (h, t) not in self.true_edge_dict:
          triples.append([ht, r])
        elif r not in self.true_edge_dict[(h, t)]:
            triples.append([ht, r])
      else:
          triples.append([ht, r])
      if len(triples) == self.num_neg:
        break

    return triples

  def __getitem__(self, idx):
    pos_sample = [self.edge_index[:, idx], self.edge_reltype[idx,:]]
    neg_samples = self._sample_negative_edge(idx, mode=self.mode)
    edges = torch.stack([pos_sample[0]] + [ht for ht, _ in neg_samples])
    edge_reltype = torch.stack([pos_sample[1]] + [r for _, r in neg_samples])
    return edges, edge_reltype

"""# Knowledge Graph Models and their Loss Functions

We define our model classes and there respective loss funtions here

**TransE**


---


TransE is based on the simple idea that the entities and relations can be seen as embeddings in a vector space such that head entity embedding and relation embedding can be added to give tail entity emebdding.

The scoring function for a positive example <h, r, t> is defined as negative of the distance, or mathematically - || h + r - t || so that distance is as low as possible for positive examples. Loss function can then be defined as a max-margin loss which maximizes the distance for negative examples and minimizes for postive examples.
"""

import torch.nn as nn
import torch.nn.functional as F

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = torch.nn.Parameter(torch.randn(num_entities, embedding_dim))
        self.relation_embeddings = torch.nn.Parameter(torch.randn(num_relations, embedding_dim))

    def forward(self):
        self.entity_embeddings.data[:-1, :].div_(
            self.entity_embeddings.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        return self.entity_embeddings, self.relation_embeddings

"""TransE Loss"""

def TransE_loss(pos_edges, neg_edges, pos_reltype, neg_reltype, entity_embeddings,
                relation_embeddings):
  # Select embeddings for both positive and negative samples
  pos_head_embeds = torch.index_select(entity_embeddings, 0, pos_edges[:, 0])
  pos_tail_embeds = torch.index_select(entity_embeddings, 0, pos_edges[:, 1])
  neg_head_embeds = torch.index_select(entity_embeddings, 0, neg_edges[:, 0])
  neg_tail_embeds = torch.index_select(entity_embeddings, 0, neg_edges[:, 1])
  pos_relation_embeds = torch.index_select(relation_embeddings, 0, pos_reltype.squeeze())
  neg_relation_embeds = torch.index_select(relation_embeddings, 0, neg_reltype.squeeze())

  # Calculate the distance score
  d_pos = torch.norm(pos_head_embeds + pos_relation_embeds - pos_tail_embeds, p=1, dim=1)
  d_neg = torch.norm(neg_head_embeds + neg_relation_embeds - neg_tail_embeds, p=1, dim=1)
  ones = torch.ones(d_pos.size(0))

  # margin loss - we want to increase d_neg and decrease d_pos
  margin_loss = torch.nn.MarginRankingLoss(margin=1.)
  loss = margin_loss(d_neg, d_pos, ones)

  return loss

"""**ComplEx**


---
ComplEx model proposes that we represent the entity and triple embeddings in a complex vector space. In ComplEx, we learn embeddings by treating the problem as a binary classification problem where the goal is to classify each triple as either positive (0) or corrupt (1).  

For a triple <h, r, t>, the similarity function takes the dot product of h, r and the complex conjugate of t and returns the real value of the product. Intuitively, this measures the similarity (specifically cosine similarity) between <h, r> and the complex conjugate of t.



"""

class ComplEx(nn.Module):
  def __init__(self, num_entities, num_relations, embedding_dim):
    super(ComplEx, self).__init__()
    self.entity_embeddings = torch.nn.Parameter(torch.randn(num_entities, embedding_dim))
    self.relation_embeddings = torch.nn.Parameter(torch.randn(num_relations, embedding_dim))

  def forward(self):
    # return the embeddings as it is but we can regularize here by normalizing them
    return self.entity_embeddings, self.relation_embeddings

"""ComplEx Loss"""

def ComplEx_loss(pos_edges, neg_edges, pos_reltype, neg_reltype,
                 entity_embeddings, relation_embeddings, reg=1e-3):
  # Select embeddings for both positive and negative samples
  pos_head_embeds = torch.index_select(entity_embeddings, 0, pos_edges[:, 0])
  pos_tail_embeds = torch.index_select(entity_embeddings, 0, pos_edges[:, 1])
  neg_head_embeds = torch.index_select(entity_embeddings, 0, neg_edges[:, 0])
  neg_tail_embeds = torch.index_select(entity_embeddings, 0, neg_edges[:, 1])
  pos_relation_embeds = torch.index_select(relation_embeddings, 0, pos_reltype.squeeze())
  neg_relation_embeds = torch.index_select(relation_embeddings, 0, neg_reltype.squeeze())

  # Get real and imaginary parts
  pos_re_relation, pos_im_relation = torch.chunk(pos_relation_embeds, 2, dim=1)
  neg_re_relation, neg_im_relation = torch.chunk(neg_relation_embeds, 2, dim=1)
  pos_re_head, pos_im_head = torch.chunk(pos_head_embeds, 2, dim=1)
  pos_re_tail, pos_im_tail = torch.chunk(pos_tail_embeds, 2, dim=1)
  neg_re_head, neg_im_head = torch.chunk(neg_head_embeds, 2, dim=1)
  neg_re_tail, neg_im_tail = torch.chunk(neg_tail_embeds, 2, dim=1)

  # Compute pos score
  pos_re_score = pos_re_head * pos_re_relation - pos_im_head * pos_im_relation
  pos_im_score = pos_re_head * pos_im_relation + pos_im_head * pos_re_relation
  pos_score = pos_re_score * pos_re_tail + pos_im_score * pos_im_tail
  pos_loss = -F.logsigmoid(pos_score.sum(1))


  # Compute neg score
  neg_re_score = neg_re_head * neg_re_relation - neg_im_head * neg_im_relation
  neg_im_score = neg_re_head * neg_im_relation + neg_im_head * neg_re_relation
  neg_score = neg_re_score * neg_re_tail + neg_im_score * neg_im_tail
  neg_loss = -F.logsigmoid(-neg_score.sum(1))

  loss = pos_loss + neg_loss
  reg_loss = reg * (
      pos_re_head.norm(p=2, dim=1)**2 + pos_im_head.norm(p=2, dim=1)**2 +
      pos_re_tail.norm(p=2, dim=1)**2 + pos_im_tail.norm(p=2, dim=1)**2 +
      neg_re_head.norm(p=2, dim=1)**2 + neg_im_head.norm(p=2, dim=1)**2 +
      neg_re_tail.norm(p=2, dim=1)**2 + neg_im_tail.norm(p=2, dim=1)**2 +
      pos_re_relation.norm(p=2, dim=1)**2 + pos_im_relation.norm(p=2, dim=1)**2 +
      neg_re_relation.norm(p=2, dim=1)**2 + neg_im_relation.norm(p=2, dim=1)**2)
  loss += reg_loss
  return loss.mean()


  # Dissect the embedding in equal chunks to get real and imaginary parts
  pos_re_head, pos_im_head = torch.chunk(pos_head_embeds, 2, dim=1)
  pos_re_tail, pos_im_tail = torch.chunk(pos_tail_embeds, 2, dim=1)
  neg_re_head, neg_im_head = torch.chunk(neg_head_embeds, 2, dim=1)
  neg_re_tail, neg_im_tail = torch.chunk(neg_tail_embeds, 2, dim=1)

  # Make phases of relations uniformly distributed in [-pi, pi]
  embedding_range = 2 * (gamma + epsilon) / pos_head_embeds.size(-1)
  pos_phase_relation = pos_relation_embeds/(embedding_range/np.pi)

  pos_re_relation = torch.cos(pos_phase_relation)
  pos_im_relation = torch.sin(pos_phase_relation)

  neg_phase_relation = neg_relation_embeds/(embedding_range/np.pi)
  neg_re_relation = torch.cos(neg_phase_relation)
  neg_im_relation = torch.sin(neg_phase_relation)


  # Compute pos score
  pos_re_score = pos_re_head * pos_re_relation - pos_im_head * pos_im_relation
  pos_im_score = pos_re_head * pos_im_relation + pos_im_head * pos_re_relation
  pos_re_score = pos_re_score - pos_re_tail
  pos_im_score = pos_im_score - pos_im_tail
  # Stack and take squared norm of real and imaginary parts
  pos_score = torch.stack([pos_re_score, pos_im_score], dim = 0)
  pos_score = pos_score.norm(dim = 0)
  # Log sigmoid of margin loss
  pos_score = gamma - pos_score.sum(dim = 1)
  pos_score = - F.logsigmoid(pos_score)

  # Compute neg score
  neg_re_score = neg_re_head * neg_re_relation - neg_im_head *neg_im_relation
  neg_im_score = neg_re_head * neg_im_relation + neg_im_head * neg_re_relation
  neg_re_score = neg_re_score - neg_re_tail
  neg_im_score = neg_im_score - neg_im_tail
  # Stack and take squared norm of real and imaginary parts
  neg_score = torch.stack([neg_re_score, neg_im_score], dim = 0)
  neg_score = neg_score.norm(dim = 0)
  # Log sigmoid of margin loss
  neg_score = gamma - neg_score.sum(dim = 1)
  neg_score = - F.logsigmoid(-neg_score)

  loss = (pos_score + neg_score)/2

  return loss.mean()

"""# Metrics and Model Evaluation

Helper routine to get the metric values given the predicted scores for a bunch of negative samples along with a positive sample that is always the first element at index 0. We currently have functionality to report these metrics:



"""

def eval_metrics(y_pred):
  argsort = torch.argsort(y_pred, dim = 1, descending = False)
  # not using argsort to do the rankings to avoid bias when the scores are equal
  ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
  ranking_list = ranking_list[:, 1] + 1
  hits5_list = (ranking_list <= 5).to(torch.float)
  hits10_list = (ranking_list <= 10).to(torch.float)
  mr_list = ranking_list.to(torch.float)

  return  hits5_list.mean(), hits10_list.mean(), mr_list.mean()

"""Evaluation routine which given a head and relation, it ranks the original positive entity along with a bunch of negative entities on the basis of scoring criteria per model and calculates above metrics"""

def eval(entity_embeddings, relation_embeddings, dataloader, kg_model, iters=None, gamma = 5.0, epsilon = 2.0):


  hits5_list = []
  hits10_list = []
  mr_list = []
  data_iterator = iter(dataloader)
  if iters is None:
    iters = len(dataloader)
  for _ in tqdm.trange(iters, desc="Evaluating"):
    batch = next(data_iterator)
    edges, edge_reltype = batch
    b, num_samples, _= edges.size()
    edges = edges.view(b*num_samples, -1)
    edge_reltype = edge_reltype.view(b*num_samples, -1)

    head_embeds = torch.index_select(entity_embeddings, 0, edges[:, 0])
    relation_embeds = torch.index_select(relation_embeddings, 0, edge_reltype.squeeze())
    tail_embeds = torch.index_select(entity_embeddings, 0, edges[:, 1])

    if kg_model == "TransE":
      scores = torch.norm(head_embeds + relation_embeds - tail_embeds, p=1, dim=1)
    elif kg_model == "ComplEx":
      # Get real and imaginary parts
      re_relation, im_relation = torch.chunk(relation_embeds, 2, dim=1)
      re_head, im_head = torch.chunk(head_embeds, 2, dim=1)
      re_tail, im_tail = torch.chunk(tail_embeds, 2, dim=1)

      # Compute scores
      re_score = re_head * re_relation - im_head * im_relation
      im_score = re_head * im_relation + im_head * re_relation
      scores = (re_score * re_tail + im_score * im_tail)
      # Negate as we want to rank scores in ascending order, lower the better
      scores = - scores.sum(dim=1)


      # Make phases of relations uniformly distributed in [-pi, pi]
      embedding_range = 2 * (gamma + epsilon) / head_embeds.size(-1)
      phase_relation = relation_embeds/(embedding_range/np.pi)
      re_relation = torch.cos(phase_relation)
      im_relation = torch.sin(phase_relation)

      # Compute scores
      re_score = re_head * re_relation - im_head * im_relation
      im_score = re_head * im_relation + im_head * re_relation
      re_score = re_score - re_tail
      im_score = im_score - im_tail
      scores = torch.stack([re_score, im_score], dim = 0)
      scores = scores.norm(dim = 0)
      scores = scores.sum(dim = 1)
    else:
      raise ValueError(f'Unsupported model {kg_model}')

    scores = scores.view(b, num_samples)

    hits5, hits10, mr = eval_metrics(scores)
    hits5_list.append(hits5.item())
    hits10_list.append(hits10.item())
    mr_list.append(mr.item())



  hits5 = sum(hits5_list)/len(hits1_list)
  hits10 = sum(hits10_list)/len(hits1_list)
  mr = sum(mr_list)/len(hits1_list)


  return  hit5, hits10, mr

"""# Training"""

#@title Choose your model and training parameters
kg_model = "TransE" #@param ["TransE", "ComplEx", "RotatE"]
epochs = 1000 #@param {type:"slider", min:10, max:1000, step:10}
batch_size = 128 #@param {type:"number"}
learning_rate = 1e-3 #@param {type:"number"}

num_entities = 14541
num_relations = 237

if kg_model == "TransE":
    model = TransE(num_entities, num_relations, 100)
    model_loss = TransE_loss
elif kg_model == "ComplEx":
    model = ComplEx(num_entities, num_relations, 100)
    model_loss = ComplEx_loss

else:
    raise ValueError('Unsupported model %s' % kg_model)

num_workers = os.cpu_count()

train_dataset = RelationDataset(train_edge, true_edges, filter=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataset = RelationDataset(valid_edge, true_edges, filter=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

val_eval_dataset = TestRelationDataset(valid_edge, true_edges, filter=True, num_neg=100)
val_eval_dataloader = DataLoader(val_eval_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataset = TestRelationDataset(test_edge, true_edges, filter=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print(f'Train dataset size {len(train_dataset)}')
print(f'Val dataset size {len(val_dataset)}')
print(f'Test dataset size {len(test_dataset)}')

# use adam optimizer for training
optimizer = torch.optim.DSGD(model.parameters(), lr=learning_rate)

for e in range(epochs):
  losses = []
  # check evaluation metrics every 10th epoch
  if e%10 == 0:
    model.eval()
    h5, h10, mr = eval(model.entity_embeddings, model.relation_embeddings, val_eval_dataloader, kg_model, iters=10)
    print(f" hits@5:{h5} hits@10:{h10} mr:{mr} ")
  model.train()
  for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc="Training")):
    # generate positive as well as negative samples for training
    pos_sample, neg_sample = batch
    # do a forward pass through the model
    entity_embeddings_pass, relation_embeddings_pass = model()

    optimizer.zero_grad()

    # compute the loss as per your model scoring criteria
    loss = model_loss(pos_sample[0], neg_sample[0], pos_sample[1], neg_sample[1],
                      entity_embeddings_pass, relation_embeddings_pass)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

  val_losses = []
  model.eval()
  entity_embeddings_pass, relation_embeddings_pass = model()
  # compute validation loss on unseen samples we didn't train on
  for step, batch in enumerate(tqdm.tqdm(val_dataloader, desc="Validating")):
    pos_sample, neg_sample = batch
    loss = model_loss(pos_sample[0], neg_sample[0], pos_sample[1], neg_sample[1],
                      entity_embeddings_pass, relation_embeddings_pass)
    val_losses.append(loss.item())

  print(f"epoch: {e + 1} loss: {sum(losses)/len(losses)} val_loss: {sum(val_losses)/len(val_losses)}")

"""Now let's test if our model actually learned something!"""

#@title Test your trained model
iterations = 1000 #@param {type:"slider", min:100, max:2000, step:100}
mode = "head" #@param ["head", "tail"]

model.eval()
test_dataset = TestRelationDataset(test_edge, true_edges, filter=True, mode=mode)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

eval(model.entity_embeddings, model.relation_embeddings, test_dataloader, kg_model, iters=iterations)

"""For reference, we tabulate the results we obtained for different models here:"""
