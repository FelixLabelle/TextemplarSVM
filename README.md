# TextemplarSVM

I saw an Andrej Karpathy post on using SVMs instead of KNN to look up similar objects. This code is a small test bench to evaluate the performance difference.

## Requirements
### Python 3.8+
Libraries: Scikit learn, Numpy, SciPY

## Experiment

Three "models" were "implemented":
1) cosine similarity
2) Linear Kernel SVM
3) Linear Kernel SVM with Normalized Vectors. I've rarely seen the L2 norm used with word embeddings, but normalizing the vectors makes the L2 norm proportional to cosine similarity (plus a constant and a scaling factor)

Three datasets were tested
1) glassgow ir
2) trec-covid (covid news corpus)
3) nfcorpus (bio-med retrieval)

Metrics reported are 
1) Sensitivty (is a result present)
2) Precision
3) Recall
4) NDCG
5) Time

## Results

The "tables" folder contains results for each experiment run. Each file name has the model type, embedding used. Each row
is a the average of a metric and each column represents a model.

I haven't run significance testing yet, but I noticed two trends:
1) Learned embeddings don't seem to benefit as much from SVM. TF-IDF and count-based embeddings benefit quite a bit across all the datasets I tested.
2) Learned embeddings don't seem to perform as well on the tasks themselves. These datasets are often used for OOD testing and I'm training tf-idf embeddings on them. I suspect that
on a more common dataset like MSMARCO this trend wouldn't hold.


## References

The repo name (and some ideas) were pulled from this blog: https://www.cs.cmu.edu/~tmalisie/projects/iccv11/
which summarizes this paper: 
Tomasz Malisiewicz, Abhinav Gupta, Alexei A. Efros. Ensemble of Exemplar-SVMs for Object Detection and Beyond . In ICCV, 2011.

https://twitter.com/karpathy/status/1647025230546886658
https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb