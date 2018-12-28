import pandas as pd
import re

comments = pd.read_csv('~/Downloads/attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('~/Downloads/attack_annotations.tsv',  sep = '\t')


def data_prep(data_frame):

new_set=[]
# len(comments.values)
for i in range(0,100):
    q=comments.values[i][0].replace("NEWLINE_TOKEN",'')
    e=q.replace("``",'')
    s=e.split('.')
    new_set.append(s)
    
