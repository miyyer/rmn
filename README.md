# Relationship Modeling Networks (RMN)
Code for model described in [Feuding Families and Former Friends: Unsupervised Learning for Dynamic Fictional Relationships](http://cs.umd.edu/~miyyer/pubs/2016_naacl_relationships.pdf) along with [a dataset of character interactions](http://cs.umd.edu/~miyyer/data/relationships.csv.gz). 
Feel free to email me at miyyer@umd.edu with any comments/problems/questions/suggestions.

### dependencies: 
- python 2, numpy, theano, lasagne
- recommended to train w/ GPU, on a 980 Ti each epoch takes 2-3 minutes

### download data and train model:
- bash run.sh (does everything for you)

### visualizing learned trajectories
- Running train_rmn.py yields three output files: the model parameters (rmn_params.pkl), the learned descriptors (descriptors.log), and the learned trajectories (trajectories.log). Before generating visualizations, you need to manually label each descriptor (each line in the descriptor file). You can do this by simply inserting your labels as the first word of each line in the descriptor file.
- After labeling the descriptors, run viz.py to generate visualizations like the ones below:
<img src="http://cs.umd.edu/~miyyer/data/ClearAndPresentDanger__Ramirez__Chavez.png" width="300">
<img src="http://cs.umd.edu/~miyyer/data/alcott-little-261__Jo__Beth.png" width="300">

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