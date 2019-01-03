import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import re
from Data_prep import *
from negationClass import *
##### 2.a Negation of attack, Toxicity and aggression ### 
### load data
##Attack
attack_comments = pd.read_csv('attack_annotated_comments.tsv', sep = '\t', index_col = 0)
attack_annotations = pd.read_csv('attack_annotations.tsv',  sep = '\t')
## Toxicity
tox_comments = pd.read_csv('toxicity_annotated_comments.tsv', sep = '\t', index_col = 0)
tox_annotations = pd.read_csv('toxicity_annotations.tsv',  sep = '\t')
## Aggression
agg_comments = pd.read_csv('aggression_annotated_comments.tsv', sep = '\t', index_col = 0)
agg_annotations = pd.read_csv('aggression_annotations.tsv',  sep = '\t')

# labels a comment as an atack if the majority of annoatators did so
attack_labels = attack_annotations.groupby('rev_id')['attack'].mean() > 0.5
tox_labels = tox_annotations.groupby('rev_id')['attack'].mean() > 0.5
agg_labels = agg_annotations.groupby('rev_id')['attack'].mean() > 0.5

# join labels and comments
attack_comments['attack'] = attack_labels
tox_comments['attack'] = tox_labels
agg_comments['attack'] = agg_labels

## take just the attack labeled sentence so to negate them:
attack=attack_comments.query('attack==True')
toxicity=tox_comments.query('attack==True')
aggression=agg_comments.query('attack==True')

### Cleaning
attack=clean_text(attack)
toxicity=clean_text(toxicity)
aggression=clean_text(aggression)

### Negating
with open('negated_attack.csv','w') as file:
    for i in range(len(dataframe)):
        line=sent_negation(dataframe.values[i][0])
        file.write(line)
        file.write('\n')

