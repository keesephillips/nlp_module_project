import os
import urllib.request
import requests
import pandas as pd
import datetime
from datetime import date
from dateutil.relativedelta import *
import time

# Replace with your New York Times API Key
with open(f'{os.getcwd()}/api_key.txt', 'r') as api:
    API_KEY = api.read()
    print(f'API Key:\t{API_KEY}')

categories = [
    'Business Day',
    'World',
    'Arts',
    'Times Insider',
    'U.S.',
    'Travel',
    'Style',
    'Food',
    'Real Estate',
    'Movies',
    'Briefing',
    'Science',
    'Your Money',
    'The Learning Network',
    'Climate',
    'Health',
    'Theater',
    'Books',
    'Magazine',
    'Sports',
    'Fashion & Style',
    'T Magazine',
    'Technology',
    'Multimedia/Photos',

]


def get_archive(path,url,filename):
  try:
    os.mkdir(path)
  except:
    path=path

  urllib.request.urlretrieve(url,f"{path}/{filename}.parquet")


if __name__ == '__main__':
    urls = {
        "train": "https://huggingface.co/datasets/stanfordnlp/sst2/resolve/main/data/train-00000-of-00001.parquet?download=true",
        "val": "https://huggingface.co/datasets/stanfordnlp/sst2/resolve/main/data/validation-00000-of-00001.parquet?download=true",
        "test": "https://huggingface.co/datasets/stanfordnlp/sst2/resolve/main/data/test-00000-of-00001.parquet?download=true"
    }
    
    for dataset in urls:
        get_archive(f'{os.getcwd()}/data/raw',urls[dataset],dataset)
        
    if not os.path.exists('/data/processed'):
        os.makedirs('/data/processed')


    # start date of 1/1/2022
    start_date = datetime.date(2023, 1, 1)
    
    # end date of today
    end_date = date.today()
    
    # delta time
    delta = relativedelta(months=1)
    
    # iterate over range of dates
    while (start_date <= end_date):
        try:
            url = f'https://api.nytimes.com/svc/archive/v1/{start_date.year}/{start_date.month}.json?api-key={API_KEY}'
            # sending get request and saving the response as response object
            r = requests.get(url = url)

            # extracting data in json format
            data = r.json()

            data = pd.DataFrame(data['response']['docs'])
            data = data[data['type_of_material'] == 'News']
            data = data[data['section_name'].isin(categories)]
            data.dropna(subset=['print_section'], inplace=True)
            data.dropna(subset=['snippet'], inplace=True)
            data = data[data['keywords'] != '[]']
            
            data.to_csv(f'./data/raw/{start_date.year}_{start_date.month}.csv')
            
            start_date += delta
            time.sleep(2)
            
        except Exception as e:
            print(e)
            time.sleep(10)
