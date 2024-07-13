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

### If you want to run the full pipeline and train the model from scratch
1. You will need to create and store a New York Times API Key within the [api_key.txt](./api_key.txt) file
2. You will need to install all of the necessary packages to run the setup.py script beforehand
3. You will then need to run setup.py to create the data pipeline and train the model
4. You will then need to run the frontend to use the model
```bash
pip install -r requirements.txt
python setup.py
streamlit run main.py
```

### If you want to just run the frontend
1. You will need to install all of the necessary packages to run the setup.py script beforehand
2. You will then need to run the frontend to use the model
```bash
pip install -r requirements.txt
streamlit run main.py
```

## Project Structure
> - requirements.txt: list of python libraries to download before running project  
> - setup.py: script to set up project (get data, train model)
> -  main.py: main script/notebook to run streamlit user interface
> - assets: directory for images used in frontend and the loss curve generated when training the model
> - scripts: directory for pipeline scripts or utility scripts  
>   - make_dataset.py: script to get data  
>   - model.py: script to train model and predict  
> - models: directory for trained models
>   - dict_model.pt: pytorch trained model for sentiment analysis stored as the state dictionary
> - data:  directory for project data
>   - raw: directory for raw data from sst2 dataset and NYT API prefetched metadata
>   - outputs: directory to store output prefetched predictions
> - notebooks: directory to store any exploration notebooks used
> - .gitignore: git ignore file

## [Data source](https://huggingface.co/datasets/stanfordnlp/sst2)
The data used to train the model was provided by Stanford University. As per their dataset description:
> The Stanford Sentiment Treebank is a corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single sentences extracted from movie reviews. It was parsed with the Stanford parser and includes a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges.

## Contributions
Brinnae Bent   
Jon Reifschneider
Falak Shah