# Claim-Detection-And-Claim-Check-Worthiness
The more we are advancing towards a modern world, the more it opens the path to falsification in every aspect of
life. Even in the case of knowing the surrounding, common people can not judge the actual scenario as the promises,
comments and opinions of the influential people in power keep changing every day. In this project we focus on solving
two problems which are part of the fact checking ecosystem that can help to automate fact-checking of claims in an ever
increasing stream of content on social media. For the first problem i.e, claim detection, we explore the fusion of syntactic
features and wordtovec, to classify whether a tweet includes a claim or not. We conduct a detailed feature analysis and present our best performing models
for English tweets.For the second problem, we check the truthfulness of the claims. We classify the claims for check-worthiness with our modified Tf-Idf model. 
We check the truthfulness of the claims by using POS, sentiment score and cosine similarity
features.
# Introduction
The concept of a claim lies at the core of the argument mining task. The difficulty with the claim detection task stems
from the disparity in conceptualization and the lack of a proper definition of a claim. The task of detecting claims
across domains has gotten a lot of attention recently, thanks to an increase in social media consumption and, by
extension, the existence of fake news, online debates, widely read blogs, and so on. For example, recent news about
the COVID-19 pandemic, with unsubstantiated claims that masks cause an increase in carbon dioxide levels, sparked
an online movement to not wear masks. With Twitter’s ease of access and sharing news, any news spreads much
faster from the moment an event occurs anywhere in the world. Despite the fact that the survey found that nearly 60
percent of users expect social media news to be inaccurate, millions of people who will spread fake news believe it to be
true. These claims can have a negative impact on individuals and society at times.In such cases, automated promotion
of claims for immediate further checks could be critical. In a nutshell claim detection can be used as a precursor to fact-checking, 
with claim segregation assisting in narrowing the corpus that requires a fact-check.
The first task is to train a model that can recognize check-worthy claims on Twitter. We present a solution that
is applicable to both English tweets. Some examples of tweets with claims are classified according to whether they are
check-worthy or not. One can see that the unsubstantiated claims appear to be personal opinions and do not pose a
threat to a larger audience. To classify the trustworthiness of a tweet, we investigate the fusion of syntactic features
and wordtovec embeddings. To compute tweet
embedding, we use POS tags, named entities, and dependency relations as syntactic features and a combination of
wordtovec. Before learning the model with a Support Vector Machine (SVM) we use Principal Component
Analysis (PCA) for dimensionality reduction.
# Motivation
Twitter, as a major Online Social Media platform, provides an ideal playground for various ideologies and perspectives.
Twitter has evolved over time and emerged as a focal point for short, unstructured pieces of text that describe anything
from current events to personal experiences in life. Most individuals view and believe things that align with their
compass and prior knowledge as known as conformity bias users tend to make bold claims that usually create a clash
between users of varied opinions. In such cases, automated promotion of claims for immediate further checks could
prove to be of utmost importance. An automated system is pivotal since Online Social Media data is far too voluminous
to allow for manual human checks, even if it was an expert.
Simultaneously, deploying separate systems based on the source of a text is inefficient and deviates from the goal of
achieving human intelligence in natural language processing tasks.
An ideal situation would be a framework that can effectively detect claims in the general setting.

# Methodology
Task-1: Tweet Claim Detection
Check-Worthiness prediction is the task of predicting whether a tweet includes a claim that is of interest to a large
audience. We approached this problem with the idea of creating a rich feature representation, reducing the dimensions
of a large feature set with PCA and then learning the model with a SVM. In doing so, our goal is also to understand
which features are the most important for check-worthiness prediction from tweet content.
Pre-processing Steps:
• Removing URLs
• Placeholders Some text cleaning was already done on the dataset which replaced some links with link and all
the videos with [video]. They don’t seem to be of any value when doing sentiment analysis so we will remove
them with regex.
• HTML reference characters were removed because they were of no use for the analysis.
• Removing punctuation
• Twitter handles were changed to “@mention” in acknowledgement of the need for protecting people’s privacy.
• Tokenizing the text
• Remove Punctuations
• Removed Emojis if any
• Removed Stop words
Syntactic Features:
We use the following syntactic features for English Parts-of-Speech (POS) tags, named entities (NE) and dependency
parse tree relations. We use the pre-processed text and run o the shelf tools to extract syntactic information of tweets
and then convert each group of information to feature sets. For English we used spaCy.
Part-of-Speech:
For both English we extract 16 POS tags in total and through our empirical evaluation we nd that the following eight
tags to be the most useful when used as features: NOUN, VERB, PROPN, ADJ, ADV, NUM, ADP, PRON.
Named Entities:
We identified the following named entity types to be the most important features through our evaluation: (GPE,
PERSON, ORG, NORP, LOC, DATE, CARDINAL, TIME, ORDINAL, FAC, MONEY). We also found that while developing
feature combinations named entities do not add much value to overall accuracy, and hence our primary and contrastive
submissions do not include them.
Syntactic Dependencies:
These features are constructed using dependency relation between tokens in a given tweet.We use the dependency
relation between two nodes in the parsed tree if the the child and parent nodes’ POS tags are one of the following ADJ,
ADV, NOUN, PROPN, VERB or NUM.
Average Word Embeddings:
One simple way to get a contextual representation of a sentence is to average the word embeddings of each token in a
given sentence. For this purpose we have used word2vec embeddings.


![image](https://user-images.githubusercontent.com/73738475/212742549-94bde3f7-c2f5-4ef8-a028-89e719331d98.png)



Figure: Syntactic feature extraction and encoding process. Feature vectors are based on the number of times it is
seen in the given sentence.


![image](https://user-images.githubusercontent.com/73738475/212742070-ba61120c-8e55-4bc7-9b9a-d27b19fbf70d.png)

Task2: Claim check Worthiness
In the Second task Tf-Idf model is created to classify the claims whether check-worthy or not. We have used a
modified Tf-Idf model with a Gradient Boosting classifier. The frequency with which a term appears in a document
or text is measured by term frequency (tf). Because each document is different in length, a term may appear more
frequently in longer documents than in shorter ones. As a result, term frequency is normalised by document length, or
the total number of terms in the document.


Idf is generally defined by the logarithm of the ratio of total number of documents in the dataset and the number
of documents with that term in them. The idf calculation is modified in our model. We Have Taken the statements
from the data as documents in Tf calculation. The modified Idf is defined as the logarithm of the ratio of total no. of
documents (the explanations from the fact-checking articles) and number of documents with the term in them. In case
of normal Tf-Idf count, the documents (or, sentences) used in both Tf and Idf belong to the same article. But in our case,
Tf count takes the actual data but Idf count takes the fact-checking news articles for consideration. The reason behind
this modification is to check word similarity between a claim and an explanation. The more the similarity is, the more
an explanation is related to the claim.

![image](https://user-images.githubusercontent.com/73738475/212743102-6ddd6ed2-f04d-4ed7-874c-a7d1aa2dfda5.png)

But in case, if the term is not present in the claim checking tweets, the Idf count is taken as 0. Now the Tf and
Idf are multiplied to calculate the modified Tf-Idf count(feature model).

![image](https://user-images.githubusercontent.com/73738475/212743281-3f4ca492-2394-477d-9878-03b3d4b8e514.png)


