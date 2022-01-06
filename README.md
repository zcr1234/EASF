########  
EASF(Attention-based Emotion-assited Sentiment Forecasting in Dialog)  
########  

========  
ABSTRACT  
Dialogue is the most basic way of communication in our lives, and it has received a lot of research, 
but there is very little research on dialogue sentiment forecasting which aims to forecast the sentimental 
polarity of what the interlocutor is about to say. Since this sentence has not been spoken, the vector of
 this sentence can’t be directly obtained. And according to cognitive science,
emotions are different from the sentiment, but there is an internal connection. 
Therefore, our paper proposes an Emotion-Assisted Sentiment Forecasting (EASF) model based on attention to
 integrating these goals. Our model uses attention to capture the significant content of emotions and sentiment, 
and emotion assistance can obtain the emotional change, then this change is used to assist in the analysis of the 
polarity of the sentiment. Experimental results show that EASF significantly outperforms all baselines.  

========  
ARCHITECTURE  
!()


The code and data of the paper(Attention-based Emotion-assited Sentiment Forecasting in Dialog)
In get_embedding, you need to download Google's pre-trained Chinese (BERT-Base, Chinese) model, put the downloaded Bert path into get_embedding, and then get the Embedding of the entire data set
After getting Embedding, add the path of data and label file to get_data
Our data set contains approximately 180,000 dialogues, and then extracts dialogues containing positive or negative sentiment. This dialogue data set contains 14,000 dialogues (including 6800 negative sentimental dialogues and 7200 positive sentimental dialogues).

The fllowing picture is the result of ten-fold cross-validation experiment:
![Result of ten-fold cross-validation experiment ](https://github.com/zcr1234/EASF/blob/main/IETGLMK%60%25KIBNOL~PSMEJ%60Y.png)

If you have any question, contacting us at 15942895821@163.com
