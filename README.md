My approach to the kaggle speech recognition challenge (https://www.kaggle.com/c/tensorflow-speech-recognition-challenge),

see my blog post at: https://dinantdatascientist.blogspot.dk/2018/02/kaggle-tensorflow-speech-recognition.html

In this repo I have implementations of a ResNet and a CTC speech-to-text model.
Start with the preproc notebook to get the training and validation arrays. Then train.ipynb.

You will need a data/, a graphs/, a logs/ and a models/ directory for the save files.

*help with the implementation of the CTC model from KerasDeepSpeech repo by @robmsmt*

![alt text](https://github.com/chrisdinant/speech/raw/master/confmat.png)
