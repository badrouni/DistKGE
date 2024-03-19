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
