# Twitter Web Scraping and Sentiment Classification via roBERTa


### Project Objective

* The purpose of this project is to review the Twitter Usage history of @elonmusk, apply the roBERTa pre-trained Transformers model for sentiment classification, and gain greater insights into the Twitter habits of Time Magazine's Person of the Year 2021.


### Methods Used

* Inferential Statistics
* Web Scraping
* Data Wrangling
* Data Preprocessing
* Data Visualization
* NLP - Sentiment Analysis



### Technologies and Packages Used

* Python and Jupyter Notebook
* Numpy, Pandas, Transformers/Huggingface, NLTK, Scipy, Pytorch
* Matplotlib, Seaborn, plotly, and more


### Project Description

* Summary: 
  - Review the Twitter usage of what some refer to as "the cyborg-like man" or the "Thomas Edison of our time", and more recently, Time Magazine's Person of the Year in 2021.
 
  
* Data and Scope:

  - The dataset was webscraped via the snscrape library, and pulled in all Tweets&Replies from Twitter user @elonmusk and consisted of Like counts, reply counts, retweet counts, datetime, and original text of tweets. Other data fields such as location were included. 
  
  
* Methodology Approach:

  - Data Preprocessing:
    1. Cleaned Twitter text for removal of stopwords, links, and punctuations.
    2. Data engineered @mentions column for list f mentioned users, as well as classificaton column for sentiment
    
  
  - Modeling Approach: 
    1. Utlized roBERTa transformer model, pre-trained on ~58 million tweets in the english language
    2. Determined sentiment classification of a tweet, by taking the sentiment score with the highest value between 0 and 1. (e.g. if a tweet has 0.82 positive score, 0.1 negative score, and 0.08 neutral score, then the tweet would be identified as being "positive")

  
### Conclusions/Findings:

  - Elon Musk has shown increased popularity within 2022 in the Twitter activity associated with his posts. This is shown in "like counts" and "reply counts". This has coincided with his take over bid of Twitter, which started in early 2022, possibly indicating that increased media attention has assisted his Twitter engagement.
  
  - Much of Elon Musk's Twitter engagement surrounds around his work with Tesla and Spacex, which demonstrates that his use of the Twitter platform is heavily leveraged as a free marketing tool for building brand equity with customers/potential future customers of his companies.
  
 
  
