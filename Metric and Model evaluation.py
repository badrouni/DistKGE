def eval_metrics(y_pred):
  argsort = torch.argsort(y_pred, dim = 1, descending = False)
  # not using argsort to do the rankings to avoid bias when the scores are equal
  ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
  ranking_list = ranking_list[:, 1] + 1
  hits5_list = (ranking_list <= 5).to(torch.float)
  hits10_list = (ranking_list <= 10).to(torch.float)
  mr_list = ranking_list.to(torch.float)
 
  return  hits5_list.mean(), hits10_list.mean(), mr_list.mean(), m

#Evaluation routine which given a head and relation, it ranks the original positive entity along with a bunch of negative entities
#on the basis of scoring criteria per model and calculates above metrics

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
  
     hits5, hits10, mr,  = eval_metrics(scores)
    hits5_list.append(hits3.item())
    hits10_list.append(hits10.item())
    mr_list.append(mr.item())
    
  
  hits5 = sum(hits3_list)/len(hits1_list)
  hits10 = sum(hits10_list)/len(hits1_list)
  mr = sum(mr_list)/len(hits1_list)
  

  return  hits5, hits10, mr
