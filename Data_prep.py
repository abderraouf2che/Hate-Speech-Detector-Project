import pandas as pd
import re

comments = pd.read_csv('~/Downloads/attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('~/Downloads/attack_annotations.tsv',  sep = '\t')

    
def clean_text(data):
    data_frame=data
    data_frame=data_frame['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
    data_frame=data_frame.apply(lambda x: x.replace("TAB_TOKEN", " "))
    data_frame=data_frame.apply(lambda x: x.lower())
    data_frame=data_frame.apply((lambda x: re.sub('[/(){}\[\]\|@,;]','',x)))
#     data_frame=data_frame.apply((lambda x: re.sub('[^0-9a-z #+_]','',x)))
    dataframe['comment'] = dataframe['comment'].apply((lambda x: re.sub('`',' ',x)))
    data_frame=data_frame.apply((lambda x: re.sub(' +',' ',x)))
    return data_frame
