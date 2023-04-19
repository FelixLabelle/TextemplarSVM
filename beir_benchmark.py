from collections import defaultdict
import json
import math
import os
from random import seed
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from tqdm.auto import tqdm


from retrieval_models import CosineSimilarityRetrieval, SVMRetrieval


# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/

# Choose datasets, currently only these are selected
# due to limited ram on my pc
datasets = ['nfcorpus', 'trec-covid-beir']

# Code to select between embeddings,
# this is done manually ATM

#TAG = 'all-MiniLM-L6-v2'
TAG = 'multi-qa-mpnet-base-dot-v1'
vectorizer = SentenceTransformer(TAG)
# NOTE: This is distinction exists because some vectorizers
# train on the corpus, in this case add progress bar and batching
corpus_embedding_func = lambda x: vectorizer.encode(x,batch_size=128,show_progress_bar =True,convert_to_numpy =True)
embedding_func = lambda x: vectorizer.encode(x)
'''
TAG = "tf_idf"
MAX_VOCAB_SIZE = 30_000
vectorizer = TfidfVectorizer(lowercase=True,max_features=MAX_VOCAB_SIZE)
# Used to train the embedding 
corpus_embedding_func = lambda x: vectorizer.fit_transform(x)
embedding_func = lambda x: vectorizer.transform(x)
'''
    
def read_jsonl(filename):
    with open(filename,'r',encoding="utf-8") as fh:
        json_lines = fh.readlines()
    
    data = [json.loads(json_line) for json_line in json_lines]
    return {datum['_id'] : datum['text'] for datum in data}

# Set random seed(s) to reduce variance in code
seed(0)
np.random.seed(0)

# Loop through selected datasets
for dataset in tqdm(datasets):
    # Intialize variables for dataset
    all_metrics = {}
    
    # Read and format data
    corpus = read_jsonl(os.path.join(dataset,'corpus.jsonl'))
    pos_to_idx_mapping = [key for key in corpus]
    queries = read_jsonl(os.path.join(dataset,'queries.jsonl'))
    relevant_document_df = pd.read_csv(os.path.join(dataset,'qrels/test.tsv'),sep='\t')
    relevant_document_df = relevant_document_df.groupby('query-id').agg({'corpus-id': list}).reset_index()
    relevant_document_mapping = dict(zip(relevant_document_df['query-id'].astype(str),relevant_document_df['corpus-id'].astype(str)))
    
    # Embed corpus
    corpus_embeddings = corpus_embedding_func([doc for doc in corpus])
    
    # Instantiate models we want to test
    cosine_sim_retrieval_model = CosineSimilarityRetrieval(embedding_func, corpus_embeddings=corpus_embeddings)
    linear_svm_retrieval_model_500 = SVMRetrieval(embedding_func, "linear", corpus_embeddings=corpus_embeddings,max_train_set_size=500)
    linear_svm_retrieval_model_1000 = SVMRetrieval(embedding_func, "linear", corpus_embeddings=corpus_embeddings,max_train_set_size=1_000)
    linear_svm_retrieval_model_2000 = SVMRetrieval(embedding_func, "linear", corpus_embeddings=corpus_embeddings,max_train_set_size=2_000)
    linear_svm_retrieval_model_5000 = SVMRetrieval(embedding_func, "linear", corpus_embeddings=corpus_embeddings,max_train_set_size=5_000)
    cosine_svm_retrieval_model_500 = SVMRetrieval(embedding_func, "linear", normalize=True, corpus_embeddings=corpus_embeddings,max_train_set_size=500)
    cosine_svm_retrieval_model_1000 = SVMRetrieval(embedding_func, "linear", normalize=True, corpus_embeddings=corpus_embeddings,max_train_set_size=1_000)
    cosine_svm_retrieval_model_2000 = SVMRetrieval(embedding_func, "linear", normalize=True, corpus_embeddings=corpus_embeddings,max_train_set_size=2_000)
    cosine_svm_retrieval_model_5000 = SVMRetrieval(embedding_func, "linear", normalize=True, corpus_embeddings=corpus_embeddings,max_train_set_size=5_000)
    
    models = [cosine_sim_retrieval_model,linear_svm_retrieval_model_500, linear_svm_retrieval_model_1000, linear_svm_retrieval_model_2000, linear_svm_retrieval_model_5000,
    cosine_svm_retrieval_model_500, cosine_svm_retrieval_model_1000, cosine_svm_retrieval_model_2000,cosine_svm_retrieval_model_5000]
    model_names = ["Cosine Sim","Linear SVM 500 sample","Linear SVM 1000 sample","Linear SVM 2000 sample","Linear SVM 5000 sample",
    "Cosine Distance SVM 500 sample","Cosine Distance SVM 1000 sample","Cosine Distance SVM 2000 sample","Cosine Distance SVM 5000 sample"]
    assert(len(models) == len(model_names))
    for model_name, retrieval_model in tqdm(zip(model_names,models)):
        metrics = []
        mean_average_metrics = defaultdict(float)
        for query_idx, relevant_document_idxs in tqdm(relevant_document_mapping.items()):
            '''
            vectorizer = TfidfVectorizer(lowercase=True,max_features=MAX_VOCAB_SIZE)
            corpus_embeddings = vectorizer.fit_transform(doc_mapping.values())
            embedding_func = lambda x: vectorizer.transform(x)
            '''
            start_time = time.time()
            query = queries[query_idx]
            _,cosine_sim_sorted_idxs = retrieval_model.query([query])
            cosine_sim_sorted_idxs = [pos_to_idx_mapping[cosine_sim_sorted_idx] for cosine_sim_sorted_idx in cosine_sim_sorted_idxs]
            run_time = time.time() - start_time
            # METRICS
            reported_metrics = {}
            for k in [1,2,5,10,20]:
                top_k_preds = cosine_sim_sorted_idxs[:k]
                tp = sum([idx in relevant_document_idxs for idx in top_k_preds])
                reported_metrics[f"precision_at_{k}"] = tp / k
                reported_metrics[f"recall_at_{k}"] = tp / len(relevant_document_idxs)
                reported_metrics[f"sensitivity_at_{k}"] = tp > 0
            dcg = sum([float(idx in relevant_document_idxs)/ math.log2(pos+2) for pos,idx in enumerate(top_k_preds)])
            idcg = sum([float(idx in relevant_document_idxs)/ math.log2(pos+2) for pos,idx in enumerate(relevant_document_idxs)])
            reported_metrics['ndcg'] = dcg/idcg
            reported_metrics['time'] = run_time 
            metrics.append(reported_metrics)
            
            # Store average
            for key,value in reported_metrics.items():
                mean_average_metrics[key] += value/len(relevant_document_mapping)
        
        pd.DataFrame(metrics).to_csv(f'results/{dataset}_{model_name}_{TAG}.csv',index=False)
        
        all_metrics[f'{model_name}'] = mean_average_metrics
        # Save results on each loop through 
        # This creates a copy of the results in case  of a crash
        results_md_table = pd.DataFrame(all_metrics.values()).T.rename({idx:key for idx,key in enumerate(all_metrics)},axis=1).to_markdown(index=True)
        with open(f"tables/{dataset}_{TAG}_results.md",'w') as fh:
            fh.write(results_md_table)