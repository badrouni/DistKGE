#@title Choose your model and training parameters
kg_model = "TransE" #@param ["TransE"]
epochs = 1000 #@param {type:"slider", min:10, max:1000, step:10}
batch_size = 128 #@param {type:"number"}
learning_rate = 1e-3 #@param {type:"number"}

num_entities = 14541
num_relations = 237

if kg_model == "TransE":
    model = TransE(num_entities, num_relations, 100)
    model_loss = TransE_loss
else:
    raise ValueError('Unsupported model %s' % kg_model)
