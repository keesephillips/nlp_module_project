"""

"""

import pandas as pd

import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from torch import optim
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import datetime
import torch
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.sequence import pad_sequences
import datetime
from datetime import date
from dateutil.relativedelta import *
import time


def train_model(model, num_epochs=4, learning_rate=2e-5):

    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Training loop
    model.to(device)

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
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        eval_accuracy = 0
        for batch in validation_dataloader:
            with torch.no_grad():

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                outputs = model(b_input_ids,
                            attention_mask=b_input_mask
                            )
                predictions = torch.argmax(outputs, dim=-1)
                eval_accuracy += (predictions == b_labels).float().mean().item()

        eval_accuracy /= len(validation_dataloader)
        print(f"Validation Accuracy: {eval_accuracy:.4f}")

    return model

def predict_sentiment(model, tokenizer, text, device=None):
    """
    Make softmax predictions on input text using the trained BERT model.

    Args:
    - model: Trained BertForSentimentClassification model
    - text: Input text string
    - device: torch.device (if None, will use CUDA if available)

    Returns:
    - predictions: Dict containing softmax probabilities and predicted class
    """
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



class BertForSentimentClassification(nn.Module):
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
    
    train = pd.read_parquet(f'{os.getcwd()}/data/raw/train.parquet')
    val = pd.read_parquet(f'{os.getcwd()}/data/raw/val.parquet')

    df = pd.concat([train, val], ignore_index=True)

    # Get the lists of sentences and their labels.
    sentences = df['sentence'].values
    labels = df['label'].values

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Print the original sentence.
    print(' Original: ', sentences[0])

    # Print the sentence split into tokens.
    print('Tokenized: ', tokenizer.tokenize(sentences[0]))

    # Print the sentence mapped to token ids.
    print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                    )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    print('Max sentence length: ', max([len(sen) for sen in input_ids]))


    # Set the maximum sequence length.
    MAX_LEN = max([len(sen) for sen in input_ids])

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    input_ids = pad_sequences(
        input_ids, 
        maxlen=MAX_LEN, 
        dtype="long",
        value=0, 
        truncating="post", 
        padding="post"
    )

    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                random_state=2018, test_size=0.2)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                random_state=2018, test_size=0.2)

    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
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

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),
                lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

    # Number of training epochs 
    epochs = 4

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0, # Default value in run_glue.py
        num_training_steps=total_steps
    )

    trained_model = train_model(
        model=model, 
        num_epochs=epochs, 
        learning_rate=2e-5
    )

    model_dir = f'{os.getcwd()}/models/'
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    filename = 'model.pt'

    # Save the entire model
    torch.save(trained_model, model_dir+filename)
    
    # start date of 1/1/2022
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
