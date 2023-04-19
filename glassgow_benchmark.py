import os

def load_data(path):
    #_____________ Read data from CISI.ALL file and store in dictinary ________________
    with open(os.path.join(path, 'CISI.ALL')) as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
 
    doc_set = {}
    doc_id = ""
    doc_text = ""

    for l in lines:
        if l.startswith(".I"):
            doc_id = l.split(" ")[1].strip() 
        elif l.startswith(".X"):
            doc_set[doc_id] = doc_text.lstrip(" ")
            doc_id = ""
            doc_text = ""
        else:
            doc_text += l.strip()[3:] + " " 
    
    #_____________ Read data from CISI.QRY file and store in dictinary ________________
    
    with open(os.path.join(path, 'CISI.QRY')) as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
          
    qry_set = {}
    qry_id = ""
    for l in lines:
        if l.startswith(".I"):
            qry_id = l.split(" ")[1].strip() 
        elif l.startswith(".W"):
            qry_set[qry_id] = l.strip()[3:]
            qry_id = ""
    
    
    #_____________ Read data from CISI.REL file and store in dictinary ________________
    
    rel_set = {}
    with open(os.path.join(path, 'CISI.REL')) as f:
        for l in f.readlines():
            qry_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0] 
            doc_id = l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1]

            if qry_id in rel_set:
                rel_set[qry_id].append(doc_id)
            else:
                rel_set[qry_id] = []
                rel_set[qry_id].append(doc_id)
    
    doc_set = {int(id):doc for (id,doc) in doc_set.items()}
    qry_set = {int(id):qry for (id,qry) in qry_set.items()}
    rel_set = {int(qid):list(map(int, did_lst)) for (qid,did_lst) in rel_set.items()}
    
    return doc_set, qry_set, rel_set
    
if __name__ == "__main__":
    from collections import defaultdict
    
    from random import seed
    from retrieval_models import CosineSimilarityRetrieval, SVMRetrieval
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    from tqdm.auto import tqdm
    from sentence_transformers import SentenceTransformer

    seed(0)
    
    doc_mapping,query_mapping,query_doc_mapping = load_data('glassgow_ir_dataset')
    
    num_mappings = len([relationship for relationships in query_doc_mapping.values() for relationship in relationships])
    assert(all([key == idx+1 for idx,key in enumerate(doc_mapping.keys())]))
    assert(all([key == idx+1 for idx,key in enumerate(query_mapping.keys())]))
    
    # Create embeddings
    MAX_VOCAB_SIZE = 30_000
    #vectorizer = CountVectorizer(lowercase=True,max_features=MAX_VOCAB_SIZE)
    '''
    vectorizer = TfidfVectorizer(lowercase=True,max_features=MAX_VOCAB_SIZE)
    corpus_embeddings = vectorizer.fit_transform(doc_mapping.values())
    embedding_func = lambda x: vectorizer.transform(x)
    '''
    model = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = model.encode([doc for doc in doc_mapping.values()],batch_size=128,show_progress_bar =True,convert_to_numpy =True)
    embedding_func = lambda x: model.encode(x)
    
    
    retrieval_model = CosineSimilarityRetrieval(embedding_func, corpus_embeddings=corpus_embeddings)
    #retrieval_model = SVMRetrieval(embedding_func, "linear", corpus_embeddings=corpus_embeddings)
    #retrieval_model = SVMRetrieval(embedding_func, "linear", normalize=True, corpus_embeddings=corpus_embeddings)
    
    query_doc_mapping = {query_idx: relevant_document_idxs for query_idx,relevant_document_idxs in query_doc_mapping.items() if relevant_document_idxs}
    metrics = []
    for query_idx, relevant_document_idxs in tqdm(query_doc_mapping.items()):
        query = query_mapping[query_idx]
        # Skip empty documents, otherwise this impacts the metrics' meaningfulness
        _,cosine_sim_sorted_idxs = retrieval_model.query([query])
        cosine_sim_sorted_idxs = cosine_sim_sorted_idxs +1
        # METRICS
        reported_metrics = {}
        for k in [1,2,5,10,20]:
            top_k_preds = cosine_sim_sorted_idxs[:k]
            tp = sum([idx in relevant_document_idxs for idx in top_k_preds])
            reported_metrics[f"precision_at_{k}"] = tp / k
            reported_metrics[f"recall_at_{k}"] = tp / len(relevant_document_idxs)
            reported_metrics[f"sensitivity_at_{k}"] = tp > 0
        metrics.append(reported_metrics)
    
    mean_average_metrics = defaultdict(float)
    for item in metrics:
        for key in metrics[0].keys():
            mean_average_metrics[key] += item[key]/len(query_doc_mapping)
    
    print(mean_average_metrics)
    # "Ablations" to run
    # 1. Embeddings (count based, tf idf, sentence embedding)
    # 2. Model (for sentence embedding)
    
    
    # 3. Sample size (for svm based models)
    
    # load models
    # run a query 
    # for each query evaluate
    # report scores
    # print(pd.DataFrame([cosine_sim_metrics, cosine_svm_metrics]).T.to_markdown())
    # report times
    # log times...