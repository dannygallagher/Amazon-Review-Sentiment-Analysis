# Predicting Sentiment of Amazon Reviews
Tom Donnelly, William Fallon, Dan Gallagher

We apply a variety of machine learning models to predict the sentiment of
of Amazon reviews using only the associated text. Specifically, we examined positive/negative as well as neutral/non-neutral sentiment, based off the review's star rating.
While others have studied this dataset for sentiment analysis, the focus often lies on the architecture of the utilized neural network, with data preprocessing often being relatively sparse.
Therefore, our goal was to examine the extent to which a more robust preprocessing method might improve our results versus the current benchmarks.
Additionally, we examined the potential impact of architectural changes, such as the use of recently developed optimizers.

We first applied a series of preprocessing methods to clean our data. Next, we implemented a series of simple models, including Naive Bayes, Random Forests, and Bigrams, to serve as a baseline for performance. Finally, we implemented and hand-tuned deep learning models, including CNN’s, RNN’s, and transformers. Once implemented, we varied several model settings, including optimizer choice, product categories, and binary vs. neutral vs. five class prediction, to analyze how our models performed in different scenarios. Our results indicated that deep learning models were significantly better than our baseline at analyzing sentiment and predicting star ratings. Of the deep learning models we trained, BERT_BASE had the best performance. 
