# ML_Prediction_Problem

Kaggle in class competiton: https://www.kaggle.com/c/how-many-shares-2020-21/overview

Implementation of different predictive models using scikit learn and mars library to predict the number of shares of an article according to 43 features (dataset 5000 articles) prediction on (2000 articles)

## Data Description

Each observation in the data is an article, represented by 43 features. You are given labels, that is, number of shares, for 5000 of these articles; your task is to predict labels for the remaining 2000 articles.

During the competition, the leaderboard will display your score on 1000 of these articles only. The final scoring will be done on the remaining 1000 articles.

## File descriptions


.<b>train.csv</b> - the training set  

.<b>train-targets.csv</b> - the training set's labels

.<b>test-validation.csv</b> - the test set  

.<b>dummy-solution.csv</b> - a sample submission file in the correct format

.<b>features-description.txt</b> - description of the features.

## Data fields

features-description.txt gives a description of each of these columns. Here are some more details about these features:

Keyword features:

*'nb_mina_mink', 'nb_mina_maxk', 'nb_mina_avek', 'nb_maxa_mink', 'nb_maxa_maxk', 'nb_maxa_avek', 'nb_avea_mink', 'nb_avea_maxk', 'nb_avea_avek'*

When the media website publishes an article, it also associates keywords with it. Let us call \[\mathcal{A}_k\] the set of articles tagged with keyword k, c(a) the number of times article a has been shared on social media, and \[\mathcal{K}_a\] the set of keywords associated with article a.

Then

\[nb\_mina\_mink(x) = \min_{k \in \mathcal{K}_x} \min_{a \in \mathcal{A}_k} c(a)\]

\[nb\_mina\_maxk(x) = \max_{k \in \mathcal{K}_x} \min_{a \in \mathcal{A}_k} c(a)\]

\[nb\_mina\_avek(x) = \frac{1}{| \mathcal{K}_x|} \sum_{k \in \mathcal{K}_x} \min_{a \in \mathcal{A}_k} c(a)\]

\[nb\_maxa\_mink(x) = \min_{k \in \mathcal{K}_x} \max_{a \in \mathcal{A}_k} c(a)\]

\[nb\_maxa\_maxk(x) = \max_{k \in \mathcal{K}_x} \max_{a \in \mathcal{A}_k} c(a)\]

\[nb\_maxa\_avek(x) = \frac{1}{| \mathcal{K}_x|} \sum_{k \in \mathcal{K}_x} \max_{a \in \mathcal{A}_k} c(a)\]

\[nb\_maxa\_mink(x) = \min_{k \in \mathcal{K}_x} \frac{1}{| \mathcal{A}_k|} \sum_{a \in \mathcal{A}_k} c(a)\]

\[nb\_maxa\_maxk(x) = \max_{k \in \mathcal{K}_x} \frac{1}{| \mathcal{A}_k|}\sum_{a \in \mathcal{A}_k} c(a)\]

\[nb\_maxa\_avek(x) = \frac{1}{| \mathcal{K}_x|} \sum_{k \in \mathcal{K}_x} \frac{1}{| \mathcal{A}_k|} \sum_{a \in \mathcal{A}_k} c(a)\]

## Topic features

'dist_topic_0', 'dist_topic_1', 'dist_topic_2', 'dist_topic_3', 'dist_topic_4'

An algorithm called Latent Dirichlet Allocation was run on the data to discover 5 latent topics characterizing the set of articles. For each article, it is then possible to compute its similarity to each of the topics, and this is what those 5 features tell us.

## Sentiment analysis features

*subj', polar', 'pp_pos_words', 'pp_neg_words', 'pp_pos_words_in_nonneutral', 'ave_polar_pos', 'min_polar_pos', 'max_polar_pos', 'ave_polar_neg', 'min_polar_neg', 'max_polar_neg', 'subj_title', 'polar_title'*

A number of features were computed using notions of subjectivity and sentiment polarity from the natural language processing field. In essence, a number of words have been annotated by English speakers, on a scale of -1.0 to 1.0, to describe how subjective they are, and to rate their polarity. These scores are retrieved for all words in the article, and the features can be computed.
