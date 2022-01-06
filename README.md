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
The architecture of whole mode(a) and EASF(b)  
![](https://github.com/zcr1234/EASF/blob/main/UYSV4%5D%40~6%40S8D4%5D%40XNSG9EI.png)  

========  
  
  
++++++++  
DATA AND SETTING  
Our data set contains approximately 180,000 dialogues, and then extracts dialogues containing positive or negative sentiment. This dialogue data set contains 14,000 dialogues (including 6800 negative sentimental dialogues and 7200 positive sentimental dialogues).The sequence of positive sentimental dialogue and negative sentimental dialogue is chaotic and disorderly.  
Then we divide the data into 10 parts, and use a ten-fold cross-validation experiment to eliminate the bias that may be caused by dividing the data.  
The initial learning rate is 0.01, and the learning rate decays to 0.8 times the original after every 3 steps.  
We use Google's Bert to embedding words, because the file is too big to upload, so we put its download link here.(https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)  

++++++++   
RESULT  
The fllowing picture is the result of ten-fold cross-validation experiment:
![Result of ten-fold cross-validation experiment ](https://github.com/zcr1234/EASF/blob/main/IETGLMK%60%25KIBNOL~PSMEJ%60Y.png)  
  
The fllowing picture is the result of experiment:  
![](https://github.com/zcr1234/EASF/blob/main/F~8B%60DNBEN9PM2K12GXIZNE.png)  
  
++++++++  

If you have any question, contacting us at 15942895821@163.com
