# Predicting Sentiment of Amazon Reviews
Tom Donnelly, William Fallon, Dan Gallagher

We apply a variety of machine learning models to predict the sentiment of Amazon reviews (downloaded from https://nijianmo.github.io/amazon/index.html) using only the associated text. Specifically, we examined positive/negative as well as neutral/non-neutral sentiment, based off the review's star rating.
While others have studied this dataset for sentiment analysis, the focus often lies on the architecture of the utilized neural network, with data preprocessing often being relatively sparse. Additionally, the literature generally studies the positive vs. negative classification problem, often eschewing the classification of neutrality.

Therefore, our goals were to examine the extent to which a more robust preprocessing method might improve our results versus the current benchmarks and to build models capable of differentiating neutral and non-neutral reviews in addition to positive and negative ones.
Additionally, we examined the potential impact of architectural changes, such as the use of recently developed optimizers like Facebook's MADGRAD.

After preprocessing, we applied both a suite of "traditional" machine learning models as well as several neural network architectures. The "traditional" models, mainly meant as a baseline, included Naive Bayes, Random Forests, and Bigrams. The neural networks included Convolutional Neural Networks, Recurrent Neural Networks, and transformers. Once implemented, we varied several model settings, including optimizer choice, product categories, and binary vs. neutral vs. five class prediction, to analyze how our models performed in different scenarios. Of the models we trained, BERT_BASE had the best performance.

All of our neural networks performed comparably to binary state-of-the art benchmarks (https://paperswithcode.com/sota), and outperformed 5-class (stars 1-5) benchmarks. This speaks to the power of domain-knowledge-based preprocessing in sentiment analysis. 

For more details, please see the pdf of the report.
