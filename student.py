#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

Group Submission: 
Anant Krishna Mahale: z5277610 
Hang Zhu: z5233612

Changes made in the student.py
--------------------------------------------------------------------------------
Preprocessing():
In this function symbols,non-ascii, numbers, white-spaces are removed. 

Stopwords:
Used different words, inbuilt library functions. It did not have any positive effect on the outcome.

wordVectors' dimention: wordVectors = GloVe(name='6B', dim=200)
Tried with all the provided values (50, 100, 200 or 300). With the value 200, the performence of the model was better
compared to other values.

convertLabel(datasetLabel):
Since the ouput label starts from 1, decreased every label by 1, which made the range [0.0 to 4.0].

convertNetOutput(netOutput):
Now, predicted label with highest probability was selected and converted back to orignal range by adding 1. 

class network(tnn.Module):
    def __init__(self)
    here, we have defined, LSTM with linear model and a regulizer tnn.dropout.

    forward()
    Initally regularization parameter is used to select the subset of the nuerons. Inorder to ignore the inputs, the input
    is passed through "tnn.utils.rnn.pack_padded_sequence" (https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch)
    Finally, through LSTM and linear nns.

class loss(tnn.Module):
    def __init__(self):
    used Cross Entropy Loss Fn since we have multiple labels (ref:https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8)    

    forward()
    calculates the entropy.

optimiser = SGD with lr = .75


why did we chose LSTM not any other models.
Before starting the project, we researched about the model that needs to be used for the text classification models. We found that most of the aticles talk
about the LSTM than any other models espcially for reviews. Hence we started off with LSTM, then tried with CNN and RNN. LSTM was performing better compared
to other 2 models. 
Links:
https://stats.stackexchange.com/questions/222584/difference-between-feedback-rnn-and-lstm-gru
https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
https://towardsdatascience.com/predicting-sentiment-of-amazon-product-reviews-6370f466fa73
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import re
#import nltk
# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')


###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

def scrub_words(text):
    """Basic cleaning of texts."""    
    # remove html markup
    text=re.sub("(<.*?>)","",text)    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)  
    #remove all other chars 
    text = re.sub('[,.";!?:\(\)-/$\'%`=><“·^\{\}_&#»«\[\]~|@、´，]+', "", text)  
    #remove whitespace
    text=text.strip()
    return text

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    cleaned_text = [scrub_words(word) for word in sample]
    final_list =' '.join(cleaned_text).split()   
    return final_list

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    return batch

#tried with different stopwords. Did not have positive effect on the result. 
stopWords = {}

 
wordVectors = GloVe(name='6B', dim=200)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """
    result = datasetLabel.long() - 1
    return result

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """
    #prediction label with the highest probability.
    pred_label = torch.argmax(netOutput, dim=1)

    # adding +1 to the output as it was deducted in the covertlabel. 
    result = (pred_label + 1).float()
    return result

###########################################################################
################### The following determines the model ####################
###########################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()
        # defining the models using above values.
        self.dropout_reg = tnn.Dropout(0.2) 
        self.lstm_model = tnn.LSTM(input_size=200, hidden_size=200, num_layers=3, batch_first=True)
        self.linear_model = tnn.Linear(in_features=200, out_features=5)
       

    def forward(self, input, length):
        processed_input_temp = self.dropout_reg(input)
        #using inbuilt function to process the paddings. 
        processed_input = tnn.utils.rnn.pack_padded_sequence(processed_input_temp, length, batch_first=True, enforce_sorted=False)
        #passing the inputs to the lstm and linear layer. 
        lstm_output, (hidden, _) = self.lstm_model(processed_input)
        final_output = self.linear_model(hidden[-1])
        return final_output

class loss(tnn.Module):
    """
    Class for creating a custom loss function, if desired.
    You may remove/comment out this class if you are not using it.
    """
    def __init__(self):
        super(loss, self).__init__()
        self.cel_function = tnn.CrossEntropyLoss()

    def forward(self, output, target):
        result = self.cel_function(output, target)
        return result

net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""
lossFunc = loss()

###########################################################################
################ The following determines training options ################
###########################################################################
trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=0.75)
