# AIPI NLP Module Project
## Developer: Keese Phillips

## About:
The purpose of this project is to accurately classify dog breeds for veterinarians. This will help vet clinics to properly stock up on supplies, medicines, and vaccinations before visits and checkups on dog breeds they will see most in a week. Some individuals do not know what breed their dog is and will inaccurately provide the vet clinic with the wrong breed. This can cause the vet clinic to buy vaccines or medication that is meant for smaller or larger-weight dogs. The project's goal is to just have the patients upload a picture of the dog and output the dog breed so that they can accurately provide the vet clinic with the correct breed. 

## How to run the project
```bash
pip install -r requirements.txt
python setup.py
streamlit run main.py
```
1. You will need to install all of the necessary packages to run the setup.py script beforehand
2. You will then need to run setup.py to create the data pipeline and train the model
3. You will then need to run the frontend to use the model

## [Data source](http://vision.stanford.edu/aditya86/ImageNetDogs/)
The data used to train the model was provided by Stanford University. With large thanks to Khosla Aditya,  Jayadevaprakash Nityananda, Yao Bangpeng, and Fei-Fei Li. 

## Contributions
Brinnae Bent   
Jon Reifschneider
