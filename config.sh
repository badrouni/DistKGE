# Configuration for TransE
# 
bash run.sh train TransE FB15k 0 0 1024 256 1000 24.0 1.0 0.0001 150000 16
bash run.sh train TransE FB15k-237 0 0 1024 256 1000 9.0 1.0 0.00005 100000 16
bash run.sh train TransE wn18 0 0 512 1024 500 12.0 0.5 0.0001 80000 8
bash run.sh train TransE malKG 0 0 512 1024 500 6.0 0.5 0.00005 80000 8
