# TextemplarSVM

I saw an Andrej Karpathy post on using SVMs instead of KNN to look up similar objects. This code is a small test bench to evaluate the differences.

The Glasgow IR dataset was used to compare Cosine Similarity and SVM over Cosine Similarity. It is small dataset with ~90 queries (that have matching documents in the corpus).

## Results

Below is a table of results comparing the Cosine SVM and Similarity. The embedding used was SentenceBert, but the results are similar for
other embeddings (e.g., CountBased and TF-IDF).


|                   |        Cosine SVM |Cosine Similarity|
|:------------------|----------:|----------:|
| precision_at_1    | 0.513158  | 0.486842  |
| recall_at_1       | 0.0343722 | 0.0240578 |
| sensitivity_at_1  | 0.513158  | 0.486842  |
| precision_at_2    | 0.5       | 0.493421  |
| recall_at_2       | 0.0553037 | 0.0404842 |
| sensitivity_at_2  | 0.657895  | 0.631579  |
| precision_at_5    | 0.447368  | 0.436842  |
| recall_at_5       | 0.108698  | 0.0793594 |
| sensitivity_at_5  | 0.763158  | 0.802632  |
| precision_at_10   | 0.398684  | 0.388158  |
| recall_at_10      | 0.164497  | 0.123798  |
| sensitivity_at_10 | 0.881579  | 0.842105  |
| precision_at_20   | 0.335526  | 0.325     |
| recall_at_20      | 0.26412   | 0.201206  |
| sensitivity_at_20 | 0.986842  | 0.907895  |
| Time (according to TQDM) | < 1 seconds | 82 seconds |

Overall the results for the cosine similarity results are better, but no significance testing was conducted (as of yet).

## Requirements
### Python 3.8+
Libraries: Scikit learn, Numpy, SciPY

## Notes

This is far from production ready, it's slow and used word count embeddings. Changing the embeddings to learned embeddings may give you a
much better bang for your buck IMO

## References

The repo name (and some ideas) were pulled from this blog: https://www.cs.cmu.edu/~tmalisie/projects/iccv11/
which summarizes this paper: 
Tomasz Malisiewicz, Abhinav Gupta, Alexei A. Efros. Ensemble of Exemplar-SVMs for Object Detection and Beyond . In ICCV, 2011.

https://twitter.com/karpathy/status/1647025230546886658
https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb