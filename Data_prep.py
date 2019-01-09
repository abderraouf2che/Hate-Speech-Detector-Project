import pandas as pd
import re

comments = pd.read_csv('~/Downloads/attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('~/Downloads/attack_annotations.tsv',  sep = '\t')

    
def clean_text(data):
	data_frame=data

	data_frame['comment']=data_frame['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
	data_frame['comment']=data_frame['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
	data_frame['comment']=data_frame['comment'].apply(lambda x: x.lower())
	data_frame['comment']=data_frame['comment'].apply((lambda x: re.sub(r'\d+', ' ',x)))
	data_frame['comment']=data_frame['comment'].apply((lambda x: re.sub('[/(){}\[\]\|@,;]',' ',x)))
    # data_frame['comment']=data_frame['comment'].apply((lambda x: re.sub('[^0-9a-z #+_]',' ',x)))

	data_frame['comment']=data_frame['comment'].apply((lambda x: re.sub('`',' ',x)))
	data_frame['comment']=data_frame['comment'].apply((lambda x: re.sub('=',' ',x)))
	data_frame['comment']=data_frame['comment'].apply((lambda x: re.sub('_',' ',x)))
	data_frame['comment']=data_frame['comment'].apply((lambda x: re.sub(':',' ',x)))

	data_frame['comment']=data_frame['comment'].apply((lambda x: re.sub(' +',' ',x)))
	return data_frame

def prep_comments(comments, annotations, name):
	# labels a comment as an atack if the majority of annoatators did so
	labels =annotations.groupby('rev_id')[name].mean() > 0.5
	
	# join labels and comments
	comments[name] = labels
	

	### Cleaning
	return clean_text(comments)
	
