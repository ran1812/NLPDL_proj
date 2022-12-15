from datasets import load_dataset
from datasets import get_dataset_split_names
from datasets import DatasetDict, Dataset, ClassLabel
import json
import numpy as np
import datasets 

def get_dataset(dataset_name, sep_token = '$',fs_num = 32):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
	fs = False
	dataset = DatasetDict()
	labels = {}
	if dataset_name[-3:] == '_fs':
		dataset_name = dataset_name[:-3]
		fs = True

	if isinstance(dataset_name,list):
		train_label = []
		test_label = []
		train_text = []
		test_text = []
		start_label = 0
		for name in dataset_name:			
			if len(train_label) != 0:
				start_label = max(train_label) + 1
			data = get_dataset(name)
			train_label += [i + start_label for i in data['train'][:]['label']]
			test_label += [i + start_label for i in data['test'][:]['label']]
			train_text += data['train'][:]['text']
			test_text += data['test'][:]['text']
		dataset['train'] = Dataset.from_dict({'text':train_text,'label':train_label})
		dataset['test'] = Dataset.from_dict({'text':test_text,'label':test_label})
		return dataset


	if dataset_name == 'bioasq':
		for split in ['train','test']:
			with open('./datasets/'+dataset_name+'/' + split+'.json',mode='r',encoding='utf-8') as f:
				data = json.load(f)
			text = []
			label = []
			question = []
			idx = []
			tot = 0
			for i in data:
				idx.append(tot)
				label.append(i['anwser'])
				question.append(i['question'])                
				text.append(i['question']+' '+sep_token.join(i['text']))
				tot += 1
			if split == 'train':
				all_label = np.unique(np.array(label))
				label_idx = np.arange(all_label.shape[0])
				labels = dict(zip(all_label, label_idx))
			for i in range(len(label)):
				label[i] = labels[label[i]]
			d = {'text':text,'label':label}
			#d = {'question':question,'text':text,'label':label,'idx':idx}
			dataset[split] = Dataset.from_dict(d)

	elif dataset_name == 'chemprot':
		labels = {}
		for split in ['train','test']:
			tmp = Dataset.from_json('./datasets/'+dataset_name+'/' + split+'.jsonl').remove_columns('metadata')
			if split == 'train':
				all_label = np.unique(np.array(tmp['label']))
				label_idx = np.arange(all_label.shape[0])
				labels = dict(zip(all_label, label_idx))
			label = []
			for i in range(len(tmp['label'])):
				label.append(labels[tmp['label'][i]])
			tmp = tmp.remove_columns('label')
			tmp = tmp.add_column('label',label)
			dataset[split] = tmp

	elif dataset_name in ['pretrain', 'data', 'pretrain_clean']:
		with open('./datasets/'+dataset_name+'.txt',mode='r',encoding='utf-8') as f:
			data = f.readlines()
		d = {'text': data}
		dataset['train'] = Dataset.from_dict(d)
	else:
		raise ValueError("Need correct dataset name")
	# your code for preparing the dataset...

	if fs == True:
		tot = dataset['train'].num_rows
		index = np.random.choice(tot,fs_num,replace = False)
		dataset_fs = dataset['train'][index]
		dataset['train'] = Dataset.from_dict(dataset_fs)

	return dataset
