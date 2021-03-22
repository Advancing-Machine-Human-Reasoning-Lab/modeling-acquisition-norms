"""
    Module for extracting word embeddings from the model in question
    Author: Antonio Laverghetta Jr.
    alaverghett@usf.edu
"""

from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
import time
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import transformers

def GetEmbeedings(model, tokenizer, dataset, word_bank_norms=True, bs=2048, dim=768, roberta=False):
    cuda = torch.device('cuda:1')
    word_embedding_mapping = {}
    word_embeddings = np.zeros((1,dim))
    if roberta:
        model = RobertaModel.from_pretrained(model, output_hidden_states=True)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
    else:
        model = BertModel.from_pretrained(model, output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained(tokenizer)

    model.to(cuda)
    model.eval()
    with torch.no_grad():
        # getting embeddings for Wordbank terms
        if word_bank_norms:
            for index, row in dataset.iterrows():
                bert_input = torch.tensor(tokenizer.encode(row['text'] + " " + row['category'])).unsqueeze(0).to(cuda)
                outputs = model(bert_input)
                hidden_states = outputs[2]
                hidden_states = torch.stack(hidden_states, dim=0)
                hidden_states = torch.squeeze(hidden_states, dim=1)
                hidden_states = hidden_states.permute(1,0,2)
                # get the final vector representation

                hidden_states = torch.mean(hidden_states[-2], dim=0)
                word_embedding_mapping[row['text'] + " " + row['category']] = hidden_states.cpu().detach().numpy()
                word_embeddings = np.append(word_embeddings, hidden_states.unsqueeze(0).cpu().detach().numpy(),axis=0)
            
            word_embeddings = np.delete(word_embeddings, 0, axis=0)
            return word_embedding_mapping, word_embeddings
        else:
            # using kuperman norms
            i = 0
            while True:
                batch_tensor = torch.Tensor().to(cuda)
                if i == len(dataset):
                    break

                text_batch = []
                if len(dataset) - i < bs:
                    batch_size = len(dataset) - i
                else:
                    batch_size = bs
                
                for index in range(i, i+batch_size):
                    text_batch.append(dataset.iloc[index]['text'])
                
                i += batch_size
                encoding = tokenizer(text_batch, return_tensors='pt', padding=True)
                input_ids = encoding['input_ids'].to(cuda)
                attention_mask = encoding['input_ids'].to(cuda)
                outputs = model(input_ids, attention_mask=attention_mask)
                
                
                hidden_states = outputs[2]
                hidden_states = torch.stack(hidden_states, dim=0)
                hidden_states = hidden_states.permute(1,2,0,3)

                for state in range(len(hidden_states)):
                    batch_tensor = torch.cat((batch_tensor,torch.mean(hidden_states[state][-2], dim=0).unsqueeze(0)))
                
                # get the final vector representation
                word_embeddings = np.append(word_embeddings, batch_tensor.cpu().detach().numpy(),axis=0)

            word_embeddings = np.delete(word_embeddings, 0, axis=0)  
            return word_embeddings