.
   |-amazon_review
   |---de               <- language code
   |-----book           <- domain
   |-------parl         <- parallel data, split by " ||| "
   |-------test         <- testing data
   |---------negative   <- negative documents
   |---------positive
   |-------train        <- training data
   |---------negative
   |---------positive
   ...

The original data was first used in Peter Prettenhofer and Benno Stein, 
``Cross-Language Text Classification using Structural Correspondence
Learning.'', Proceedings of the ACL, 2010. 

If you use this dataset, please cite the above paper.

Please see the link for more information: https://www.uni-weimar.de/en/media/chairs/computer-science-and-media/webis/corpora/corpus-webis-cls-10/.