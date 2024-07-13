"""
Attribution: https://colab.research.google.com/gist/aicrowd-bot/eb3d8d17c5ee33efe841e2f341630c66

Falak Shah

-------------------------------------------------------------------------------

Attribution: https://github.com/AIPI540/AIPI540-Deep-Learning-Applications/

Jon Reifschneider
Brinnae Bent 

"""

import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import torch
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.sequence import pad_sequences
import datetime
from datetime import date
from dateutil.relativedelta import *
import time
import matplotlib.pyplot as plt



def prepare_data(sentences):
    '''
    Transforms the text into an input the BERT model will accept

    Inputs:
        sentences: the text data to tokenize and pad and transform

    Returns:
        tokenizer: BERT tokenizer used to transform data
        input_ids: the tokenized ids
        attention_masks: the mask for the tokens
    '''
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(
                sent,                      # Sentence to encode.
                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        )
        input_ids.append(encoded_sent)

    # Set the maximum sequence length.
    MAX_LEN = max([len(sen) for sen in input_ids])

    # Pad tokens with 0s
    input_ids = pad_sequences(
        input_ids, 
        maxlen=MAX_LEN, 
        dtype="long",
        value=0, 
        truncating="post", 
        padding="post"
    )

    attention_masks = []

    for sent in input_ids:

        # Create the attention mask.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    
    return tokenizer, input_ids, attention_masks



def train_model(model, num_epochs=4, learning_rate=2e-5):
    '''
    Trains the BERT model on the training dataset and will then calculate the 
    accuracy score on the validation dataset

    Inputs:
        model: BERT model to train
        num_epochs(int): number of epochs to use in the training
        learning_rate(float): the learning rate to use in the training
    Returns:
        model: the trained BERT model on the training set

    '''

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Training loop
    model.to(device)
    
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(b_input_ids,
                    attention_mask=b_input_mask
                    )
            loss = nn.CrossEntropyLoss()(outputs, b_labels)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        eval_accuracy = 0
        total_val_loss = 0
        for batch in validation_dataloader:
            with torch.no_grad():

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                outputs = model(b_input_ids,
                            attention_mask=b_input_mask
                            )
                loss = nn.CrossEntropyLoss()(outputs, b_labels)
                total_val_loss += loss.item()
                predictions = torch.argmax(outputs, dim=-1)
                eval_accuracy += (predictions == b_labels).float().mean().item()

        avg_val_loss = total_val_loss / len(validation_dataloader)
        val_losses.append(avg_val_loss)
        eval_accuracy /= len(validation_dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {eval_accuracy:.4f}")
        
    # Plot the validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    return model

def predict_sentiment(model, tokenizer, text, device=None):
    '''
    Make softmax predictions on input text using the trained BERT model.

    Inputs:
        model: trained BERT model
        tokenizer: the tokenizer to use to transform the data
        text: input text string
        device: torch.device (if None, will use CUDA if available)

    Returns:
        predictions: dictionary containing softmax probabilities and predicted class
    '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    # Tokenize input text
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Move input to device
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs, dim=1)

    # Convert to numpy for easier handling
    probabilities = probabilities.cpu().numpy()[0]

    if len(probabilities) == 2:
        # Binary classification
        return [float(probabilities[0]), float(probabilities[1]), float(probabilities[1]) - float(probabilities[0])]
    else:
        return [0., 0., 0.]


def generate_predictions(trained_model, tokenizer):
    '''
    Prefetches the predictions from the API to save on requests to their server and 
    save on compute needed to generate the predictions 

    Inputs:
        trained_model: trained BERT model
        tokenizer: the tokenizer to use to transform the data

    Returns:
    '''
    # start date of 1/1/2023
    start_date = datetime.date(2023, 1, 1)
    
    # end date of today
    end_date = date.today()
    
    # delta time
    delta = relativedelta(months=1)
    
    # iterate over range of dates
    while (start_date <= end_date):
        data = pd.read_csv(f'{os.getcwd()}/data/raw/{start_date.year}_{start_date.month}.csv')
        print(start_date)
        rows = []
        for index, row in data.iterrows():
            row_dict = {}
            results = predict_sentiment(trained_model, tokenizer, row['snippet'])
            row_dict['snippet'] = row['snippet']
            row_dict['negative'] = results[0]
            row_dict['positive'] = results[1]
            row_dict['sentiment_score'] = results[2]
            rows.append(row_dict)

        df = pd.DataFrame(rows)
        df = data.merge(df, left_on='snippet', right_on='snippet', how='left')
        df = df[['snippet','keywords','news_desk','headline','keywords','pub_date','negative','positive','sentiment_score']]
        df.to_csv(f'{os.getcwd()}/data/output/{start_date.year}_{start_date.month}.csv')
        
        start_date += delta



class BertForSentimentClassification(nn.Module):
    '''
    The pretrained-BERT model with an additional Linear Layer 

    '''
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2):
        super(BertForSentimentClassification, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    model_dir = '/models/'
    model_name = 'dict_model.pt'
    
    fullpath = os.getcwd()+ model_dir + model_name
    print(fullpath)
    
    if os.path.exists(fullpath):
        print("Model Found")
        trained_model = BertForSentimentClassification()
        trained_model.load_state_dict(torch.load(fullpath, map_location=device))

        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in trained_model.state_dict():
            print(param_tensor, "\t", trained_model.state_dict()[param_tensor].size())
            
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            
    else:
        print("Model Not Found")
        train = pd.read_parquet(f'{os.getcwd()}/data/raw/train.parquet')
        val = pd.read_parquet(f'{os.getcwd()}/data/raw/val.parquet')

        df = pd.concat([train, val], ignore_index=True)

        sentences = df['sentence'].values
        labels = df['label'].values
        
        tokenizer, input_ids, attention_masks = prepare_data(sentences)

        # Use 80% for training and 20% for validation.
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                    random_state=2018, test_size=0.2)
        # Do the same for the masks.
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                    random_state=2018, test_size=0.2)

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        batch_size = 32

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        model = BertForSentimentClassification()

        # Binary Classification
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            model.parameters(),
            lr = 2e-5, 
            eps = 1e-8 
        )

        # Number of training epochs 
        epochs = 4

        # Total number of training steps is number of batches * number of epochs
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )

        trained_model = train_model(
            model=model, 
            num_epochs=epochs, 
            learning_rate=2e-5
        )

        os.makedirs(os.path.dirname(os.getcwd() + model_dir), exist_ok=True)

        # Save the model's learned parameters (state_dict)
        torch.save(trained_model.state_dict(), fullpath)
    
    generate_predictions(trained_model, tokenizer)
    
    
