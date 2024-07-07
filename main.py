"""
Attribution: https://github.com/AIPI540/AIPI540-Deep-Learning-Applications/

Jon Reifschneider
Brinnae Bent 

"""


import streamlit as st
from PIL import Image
import numpy as np
import os
import numpy as np
import pandas as pd
import pandas as pd
import json
import matplotlib.pyplot as plt

def replace_json(x):
    x = x.replace("'", "\"")
    try:
        return json.loads(x)
    except:
        return None

def load_data(year, month, top_selection, category):
    df = pd.read_csv(f'data/processed/{year}_{month}.csv')

    if category != 'All':
        df = df[df['news_desk'] == category]

    df = df.sort_values(by='sentiment_score', ascending=False)
    df['sentiment_score'] = round(df['sentiment_score'].astype(float), 4)
    df['positive'] = df['positive'].astype(float)
    df['negative'] = df['negative'].astype(float)
    df['keywords'] = df['keywords'].apply(lambda x: replace_json(x))
    
    good_df = df.head(top_selection)
    bad_df = df.tail(top_selection)
    
    good_df = good_df.explode('keywords')
    bad_df = bad_df.explode('keywords')
    
    good_df = pd.concat([good_df.reset_index(drop=True), pd.json_normalize(good_df['keywords'])], axis=1)
    bad_df = pd.concat([bad_df.reset_index(drop=True), pd.json_normalize(bad_df['keywords'])], axis=1)
    
    good_df.drop(columns=['keywords'], inplace=True)
    bad_df.drop(columns=['keywords'], inplace=True)
    
    good_df.dropna(subset=['value'], inplace=True)
    bad_df.dropna(subset=['value'], inplace=True)
    
    
    good_df = good_df.drop_duplicates(subset=['value'])[['snippet','value','positive', 'sentiment_score']]
    bad_df = bad_df.drop_duplicates(subset=['value'])[['snippet','value','negative','sentiment_score']]

    return good_df, bad_df

categories = [
    'All', 'Sports', 'Foreign', 'Styles', 'Washington', 'Business', 'Science',
    'Culture', 'SundayBusiness', 'Arts&Leisure', 'Express', 'National',
    'Climate', 'Magazine', 'RealEstate', 'Politics', 'Weekend',
    'Dining', 'Books', 'BookReview', 'Travel', 'NYTNow', 'Summary',
    'Insider', 'SpecialSections', 'Investigative', 'Video', 'Live',
    'TStyle', 'Weather', 'Projects and Initiatives', 'Metro', 'Obits',
    'Graphics', 'Local Investigations', 'Election Analytics',
    'Photo', 'Headway', 'Games', 'Metropolitan'
]

months = {
    "Jan 2023": {"month": "1", "year": "2023"},
    "Feb 2023": {"month": "2", "year": "2023"},
    "Mar 2023": {"month": "3", "year": "2023"},
    "Apr 2023": {"month": "4", "year": "2023"},
    "May 2023": {"month": "5", "year": "2023"},
    "Jun 2023": {"month": "6", "year": "2023"},
    "Jul 2023": {"month": "7", "year": "2023"},
    "Aug 2023": {"month": "8", "year": "2023"},
    "Sep 2023": {"month": "9", "year": "2023"},
    "Oct 2023": {"month": "10", "year": "2023"},
    "Nov 2023": {"month": "11", "year": "2023"},
    "Dec 2023": {"month": "12", "year": "2023"},
    "Jan 2024": {"month": "1", "year": "2024"},
    "Feb 2024": {"month": "2", "year": "2024"},
    "Mar 2024": {"month": "3", "year": "2024"},
    "Apr 2024": {"month": "4", "year": "2024"},
    "May 2024": {"month": "5", "year": "2024"},
    "Jun 2024": {"month": "6", "year": "2024"},
}
    

if __name__ == '__main__':

    
    st.header('New York Times Sentiment Analysis', divider='red')
    
    col1, col2 = st.columns(2)

    emotions = Image.open('assets/emotions.png')
    col1.image(emotions, use_column_width=True)

    newspaper = Image.open('assets/newspapers.png')
    col2.image(newspaper, use_column_width=True)
    
    st.divider()
    

    # Using "with" notation
    with st.sidebar:
        month_selection = st.selectbox(
            "Month to analyze",
            (   months.keys()            )
        )
        selection = st.selectbox(
            "Top selection",
            (   np.arange(25, 50, 5)           )
        )
        category = st.selectbox(
            "Category selection",
            (   categories         )
        )
        good_df, bad_df = load_data(months[month_selection]['year'], months[month_selection]['month'], selection, category)
    
    st.subheader(f"Best Sentiment for {month_selection}")
    st.bar_chart(data=good_df, y='sentiment_score', x='value', color='#064b38')
    st.dataframe(data=good_df[['snippet', 'sentiment_score']].drop_duplicates(), hide_index=True)
    st.subheader(f"Worst Sentiment for {month_selection}")
    st.bar_chart(data=bad_df, y='sentiment_score', x='value', color='#e23a08')
    st.dataframe(data=bad_df[['snippet', 'sentiment_score']].drop_duplicates(), hide_index=True)
