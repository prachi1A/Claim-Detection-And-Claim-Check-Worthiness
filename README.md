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


