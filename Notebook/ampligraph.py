import sys
sys.path.append('../..')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import ampligraph
# Benchmark datasets are under ampligraph.datasets module
from ampligraph.datasets import load_fb15k_237, load_fb15k, load_WN18,load_MalKG
# load  datasets
dataset = load_fb15k_237()
dataset = load_fb15k()
dataset = load_WN18()
dataset = load_MalKG()

# Import the KGE model
from ampligraph.latent_features import ScoringBasedEmbeddingModel

# you can continue training from where you left after restoring the model
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./transe_train_logs')

# create the model with transe scoring function
model = ScoringBasedEmbeddingModel(eta=5,
                                   k=5,
                                   scoring_type='TransE')

# you can either use optimizers/regularizers/loss/initializers with default values or you can 
# import it and customize the hyperparameters and pass it to compile

# Let's create an adam optimizer with customized learning rate =0.005
adam = tf.keras.optimizers.Adam(learning_rate=0.005)
# Let's compile the model with self_advarsarial loss of default parameters
model.compile(optimizer=adam, loss='self_adversarial')

# fit the model to data.
model.fit(dataset['train'],
             batch_size=10000,
             epochs=1000,
             callbacks=[tensorboard_callback])

# the training can be visualised using the following command:
# tensorboard --logdir='./transe_train_logs' --port=8891 
# open the browser and go to the following URL: http://127.0.0.1:8891/

pred = model.predict(dataset['test'][:5], 
                       batch_size=100)
pred
# evaluate on the test set
ranks = model.evaluate(dataset['test'],     # test set
                       batch_size=100,      # evaluation batch size
                       corrupt_side='s,o'   # sides to corrupt for scoring and ranking
                       )

# import the evaluation metrics
from ampligraph.evaluation.metrics import  hits_at_n_score, mr_score

print('MR:', mr_score(ranks))
print('hits@5:', hits_at_n_score(ranks, 5))
print('hits@10:', hits_at_n_score(ranks, 10))
