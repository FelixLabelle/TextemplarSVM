import random

import numpy as np
from scipy.sparse import vstack
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


random.seed(0)

with open('babylm_dev_v1.dev','r',encoding="utf-8") as fh:
    search_corpus = list(map(lambda x: x.strip(), fh.readlines()))

TOP_K = 20

# Create embeddings
MAX_VOCAB_SIZE = 30_000
vectorizer = CountVectorizer(lowercase=True,max_features=MAX_VOCAB_SIZE)
corpus_embeddings = vectorizer.fit_transform(search_corpus)

query = search_corpus[0]
query_embedding = vectorizer.transform([query])

# L2 SVM
model = LinearSVC(class_weight='balanced',verbose=False, max_iter=10000, tol=1e-6, C=0.1)

# Format data
concat_query_corpus_embeddings = vstack([query_embedding, corpus_embeddings])

labels = np.zeros(len(search_corpus)+1)
labels[0] = 1

model.fit(concat_query_corpus_embeddings,labels)
similarities = model.decision_function(corpus_embeddings)
sorted_idx = np.argsort(-similarities)
l2_svm_topk = [search_corpus[i] for i in sorted_idx[:TOP_K]]

# QUESTION: What are the effects of having the query in the dataset? For SVM does
# this cause issues?
# https://www.cs.cmu.edu/~tmalisie/projects/iccv11/
# Is this a problem?
# Cosine Similarity
similarities = cosine_similarity(query_embedding, corpus_embeddings)
cosine_sim_sorted_idxs = (-similarities).squeeze(0).argsort()
cosine_sim_topk = [search_corpus[idx] for idx in cosine_sim_sorted_idxs[:TOP_K]]

# Cosine Similarity SVM

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

print("L2 SVM results")
print(l2_svm_topk)
print("Cosine SVM results")
print(cosine_sim_topk)
print("Cosine Kernel SVM results")
print(cosine_svm_topk)
print("Cosine Distance SVM results")
print(cosine_distance_svm_topk)