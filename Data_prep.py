import pandas as pd
import re

comments = pd.read_csv('~/Downloads/attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('~/Downloads/attack_annotations.tsv',  sep = '\t')


def data_prep(data_frame):
    data=data_frame
    new_set=[]
    # len(data.values)
    for i in range(0,len(data.values)):
        q=data.values[i][0].replace("NEWLINE_TOKEN",'')
        e=q.replace("``",'')
        s=e.split('.')
        new_set.append(s)
    return new_set
    
def clean_text(data_frame):
    
    data_frame=data_frame['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    data_frame=data_frame.apply(lambda x: x.replace("TAB_TOKEN", " "))
    data_frame=data_frame.apply(lambda x: x.lower())
    data_frame=data_frame.apply((lambda x: re.sub('[/(){}\[\]\|@,;]','',x)))
    data_frame=data_frame.apply((lambda x: re.sub('[^0-9a-z #+_]','',x)))
    data_frame=data_frame.apply((lambda x: re.sub('[^0-9a-z #+_]','',x)))
    data_frame=data_frame.apply((lambda x: re.sub(' +',' ',x)))
    return data_frame
