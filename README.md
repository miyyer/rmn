# Relationship Modeling Networks (RMN)
Code for model described in [Feuding Families and Former Friends: Unsupervised Learning for Dynamic Fictional Relationships](http://cs.umd.edu/~miyyer/pubs/2016_naacl_relationships.pdf) along with [a dataset of character interactions](http://cs.umd.edu/~miyyer/data/relationships.csv.gz). 
Feel free to email me at miyyer@umd.edu with any comments/problems/questions/suggestions.

### dependencies: 
- python 2, numpy, theano, lasagne
- recommended to train w/ GPU, on a 980 Ti each epoch takes 2-3 minutes

### dataset description:
- 20,046 relationships with 387,614 total spans from 1,378 different books
- each span is provided in a bag-of-words format where stopwords and infrequent words have been filtered out as described in the paper

### download data and train model:
- bash run.sh (downloads character interaction dataset, metadata info, and 300d GloVe embeddings pretrained on the Common Crawl, and then runs train_rmn.py to train an RMN on the downloaded dataset)

### visualizing learned trajectories
- Running train_rmn.py yields three output files: the model parameters (rmn_params.pkl), the learned descriptors (descriptors.log), and the learned trajectories (trajectories.log). Before generating visualizations, you need to manually label each descriptor (each line in the descriptor file). You can do this by simply inserting your labels as the first word of each line in the descriptor file.
- After labeling the descriptors, run viz.py to generate visualizations like the ones below:
<img src="http://cs.umd.edu/~miyyer/data/ClearAndPresentDanger__Ramirez__Chavez.png" width="400">
<img src="http://cs.umd.edu/~miyyer/data/alcott-little-261__Jo__Beth.png" width="400">

if you use this code, please cite:

	@inproceedings{IyyerRelationships,
		Author = {Mohit Iyyer and Anupam Guha and Snigdha Chaturvedi and Jordan Boyd-Graber and Hal {Daum\'{e} III}},
		Booktitle = {North American Association for Computational Linguistics},
		Location = {San Diego, CA},
		Year = {2016},
		Title = {Feuding Families and Former Friends: Unsupervised Learning for Dynamic Fictional Relationships},
	}

### to-dos: 
- clean and integrate the alpha tuning code
- better comment RMN hyperparams and add argparse 
