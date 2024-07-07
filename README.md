# AIPI NLP Module Project
## Developer: Keese Phillips

## About:
The purpose of this project is to determine the sentiment for topics throughout the past year and a half. The model 
ingests data via the New York Times API and employs the model to determine the sentiment for the topics within
the articles released on a monthly basis. The model will display the best and worst sentiment for articles
throughout the month and will display the topics associated within the article. The model will also display the 
sentiment score for the article/topics which is calculated as the difference between the positive sentiment
score and the negative sentiment score for the associated article/topic.

## How to run the project
```bash
pip install -r requirements.txt
python setup.py
streamlit run main.py
```
1. You will need to create and store a New York Times API Key within the api_key.txt file
2. You will need to install all of the necessary packages to run the setup.py script beforehand
3. You will then need to run setup.py to create the data pipeline and train the model
4. You will then need to run the frontend to use the model

## [Data source](https://huggingface.co/datasets/stanfordnlp/sst2)
The data used to train the model was provided by Stanford University. As per their dataset description:
> The Stanford Sentiment Treebank is a corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single sentences extracted from movie reviews. It was parsed with the Stanford parser and includes a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges.

## Contributions
Brinnae Bent   
Jon Reifschneider
