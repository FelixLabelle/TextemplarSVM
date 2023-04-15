# TextemplarSVM

I saw an Andrej Karpathy post on using SVMs instead of KNN to look up similar objects. This code is a small test bench to (subjectively) evaluate the differences.

The Linear SVM does not work well, but this is expected IMO. L2 norm is rarely (if ever) used when comparing word/language embeddings.

There are two cosine based methods, one uses a cosine similarity kernel and the other uses precomputed cosine similarities. Both give
better results than linear SVM, but I haven't benchmarked these models so no clear winner. 

Subjectively I like the precomputed cosine SVM most

## Requirements
### Python
Libraries: Scikit learn, Numpy, SciPY
Version 3.8+

## Notes

This is far from production ready, it's slow and used word count embeddings. Changing the embeddings to learned embeddings may give you a
much better bang for your buck IMO

## References

The repo name (and some ideas) were pulled from this blog: https://www.cs.cmu.edu/~tmalisie/projects/iccv11/
which summarizes this paper: 
Tomasz Malisiewicz, Abhinav Gupta, Alexei A. Efros. Ensemble of Exemplar-SVMs for Object Detection and Beyond . In ICCV, 2011.

https://twitter.com/karpathy/status/1647025230546886658
https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb