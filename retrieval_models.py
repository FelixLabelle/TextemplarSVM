import random

import numpy as np
from scipy.sparse import vstack
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics.pairwise import cosine_similarity

'''
# Format data
l2_svm_topk = [search_corpus[i] for i in sorted_idx[:TOP_K]]

# QUESTION: What are the effects of having the query in the dataset? For SVM does
# this cause issues?
# https://www.cs.cmu.edu/~tmalisie/projects/iccv11/
# Is this a problem?
# Cosine Similarity
'''
class CosineSimilarityRetrieval:
    def __init__(self, vectorizer,corpus=None,corpus_embeddings=None):
        self.vectorizer = vectorizer
        if type(corpus_embeddings) != None:
            self.corpus_embeddings = corpus_embeddings
        elif type(corpus) != None:
            corpus_embeddings = self.vectorizer(corpus)
            self.corpus_embeddings = corpus_embeddings
        else:
            raise ValueError("No corpus or corpus embeddings passed in")
    
    def query(self, query_doc):
        query_embedding = self.vectorizer(query_doc)
        similarities = cosine_similarity(query_embedding, self.corpus_embeddings)
        sorted_idx = (-similarities).squeeze(0).argsort()
        return similarities, sorted_idx

class SVMRetrieval:
    def __init__(self, vectorizer,kernel="linear",corpus=None,corpus_embeddings=None, normalize=False, max_train_set_size=-1):
        self.vectorizer = vectorizer
        self.kernel = kernel
        self.normalize = normalize
        
        if type(corpus_embeddings) != None:
            self.corpus_embeddings = corpus_embeddings
        elif type(corpus) != None:
            corpus_embeddings = self.vectorizer(corpus)
            self.corpus_embeddings = corpus_embeddings
        else:
            raise ValueError("No corpus or corpus embeddings passed in")
        
        if self.normalize:
            self.corpus_embeddings = sk_normalize(self.corpus_embeddings)
            
        # todo: pass in HPs
        if max_train_set_size > 0:
            sample_idxs = random.sample(range(0,self.corpus_embeddings.shape[0]),k=min(self.corpus_embeddings.shape[0],max_train_set_size))
            self.train_embeddings = self.corpus_embeddings[sample_idxs,:]
        else:
            self.train_embeddings = self.corpus_embeddings
       
        if self.kernel == "linear":
            self.model = LinearSVC(class_weight='balanced',verbose=False, max_iter=10000, tol=1e-6, C=0.1)
        elif self.kernel == "precomputed":
            self.model = SVC(kernel="precomputed",class_weight='balanced',verbose=False, max_iter=10000, tol=1e-6, C=0.1)
        elif self.kernel == "cosine_sim":
            self.model = SVC(kernel=cosine_similarity,class_weight='balanced',verbose=False, max_iter=10000, tol=1e-6, C=0.1)
        else:
            raise NotImplementedError(f"Kernel {kernel} is not currently supported")
            
        self.labels = np.zeros(self.train_embeddings.shape[0]+1)
        self.labels[0] = 1
        
    def query(self, query_doc):
        query_embedding = self.vectorizer(query_doc)
        
        if self.normalize:
            query_embedding = sk_normalize(query_embedding)
            
        concat_query_corpus_embeddings = vstack([query_embedding, self.train_embeddings])
        
        if self.kernel == "precomputed":
            training_embeddings = cosine_similarity(concat_query_corpus_embeddings)
        else:
            training_embeddings = concat_query_corpus_embeddings
            
        self.model.fit(training_embeddings,self.labels)
        
        if self.kernel == "precomputed":
            corpus_embeddings = cosine_similarity(self.corpus_embeddings,concat_query_corpus_embeddings)
        else:
            corpus_embeddings = self.corpus_embeddings
            
        similarities = self.model.decision_function(corpus_embeddings)
        sorted_idx = np.argsort(-similarities)
        return similarities, sorted_idx
        
# Cosine Similarity SVM
'''
model = SVC(kernel=cosine_similarity,class_weight='balanced',verbose=False, max_iter=10000, tol=1e-6, C=0.1)
SVM_SAMPLE_SIZE = 1000
sampled_idxs = [0] + random.sample(range(1,concat_query_corpus_embeddings.shape[0]),k=SVM_SAMPLE_SIZE)
model.fit(concat_query_corpus_embeddings[sampled_idxs,:],labels[sampled_idxs])
similarities = model.decision_function(corpus_embeddings)
sorted_idx = np.argsort(-similarities)
cosine_svm_topk = [search_corpus[i] for i in sorted_idx[:TOP_K]]

print(l2_svm_topk)
print(cosine_sim_topk)
print(cosine_svm_topk)

# Cosine Similarity SVM
model = SVC(kernel="precomputed",class_weight='balanced',verbose=False, max_iter=10000, tol=1e-6, C=0.1)
SVM_SAMPLE_SIZE = 1000
sampled_idxs = [0] + random.sample(range(1,concat_query_corpus_embeddings.shape[0]),k=SVM_SAMPLE_SIZE)
cosine_sim_distances  = cosine_similarity(concat_query_corpus_embeddings[sampled_idxs,:])
model.fit(cosine_sim_distances,labels[sampled_idxs])
dev_sim = cosine_similarity(corpus_embeddings,concat_query_corpus_embeddings[sampled_idxs,:])
similarities = model.decision_function(dev_sim)
sorted_idx = np.argsort(-similarities)
cosine_distance_svm_topk = [search_corpus[i] for i in sorted_idx[:TOP_K]]
'''
if __name__ == "__main__":
    from sklearn.feature_extraction.text import CountVectorizer
    random.seed(0)
    
    with open('babylm_dev_v1.dev','r',encoding="utf-8") as fh:
        search_corpus = list(map(lambda x: x.strip(), fh.readlines()))

    TOP_K = 20

    # Create embeddings
    MAX_VOCAB_SIZE = 30_000
    vectorizer = CountVectorizer(lowercase=True,max_features=MAX_VOCAB_SIZE)
    corpus_embeddings = vectorizer.fit_transform(search_corpus)
    
    # NOTE: Make sure the query is the format expected by your vectorizer
    query = [search_corpus[0]]

    cosine_sim_search = CosineSimilarityRetrieval(lambda x: vectorizer.transform(x), corpus_embeddings=corpus_embeddings)
    _,cosine_sim_sorted_idxs = cosine_sim_search.query(query)
    cosine_sim_topk = [search_corpus[idx] for idx in cosine_sim_sorted_idxs[:TOP_K]]

    l2_svm_search = SVMRetrieval(lambda x: vectorizer.transform(x), "linear", corpus_embeddings=corpus_embeddings)
    _,l2_svm_sorted_idxs = l2_svm_search.query(query)
    l2_svm_topk = [search_corpus[idx] for idx in l2_svm_sorted_idxs[:TOP_K]]

    cosine_svm_search = SVMRetrieval(lambda x: vectorizer.transform(x), "cosine_sim", corpus_embeddings=corpus_embeddings,sample_size=1000)
    _,cosine_svm_sorted_idxs = cosine_svm_search.query(query)
    cosine_svm_topk = [search_corpus[idx] for idx in cosine_svm_sorted_idxs[:TOP_K]]
    
    cosine_distance_svm_search = SVMRetrieval(lambda x: vectorizer.transform(x), "linear", corpus_embeddings=corpus_embeddings,normalize=True,sample_size=1000)
    _,cosine_distance_svm_sorted_idxs = cosine_distance_svm_search.query(query)
    cosine_distance_svm_topk = [search_corpus[idx] for idx in cosine_distance_svm_sorted_idxs[:TOP_K]]
    
    print("L2 SVM results")
    print(l2_svm_topk)
    print("Cosine SVM results")
    print(cosine_sim_topk)
    #print("Cosine Kernel SVM results")
    #print(cosine_svm_topk)
    print("Cosine Distance SVM results")
    print(cosine_distance_svm_topk)


