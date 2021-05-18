# tec_forecast
Tensorflow implementation of a TEC forecast. Similar to (Boulch, 2018)

BOULCH, A.; CHERRIER, N.; CASTAINGS, T. Ionospheric activity prediction using convolutional recurrent neural networks. arXiv:1810.13273 [cs], 6 nov. 2018. 
https://github.com/aboulch/tec_prediction/



## download 3 years of data (2018-2020)
python3 downloader.py 

## create numpy representation for the data downloaded
python3 ionex_samples.py 

## Training the network
python3 tec_lstm.py

The parameters.py file is created during training. If you retrain the network, please remove it.
