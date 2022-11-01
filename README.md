# DeepTEC
A deep learning laboratory for Total Electron Content prediction experiments, using Tensorflow 2.\
Models can be found at ./models/models.py. Some reusable layers are on ./models/custom_layers.py.\
The c111 and c333 models were inspired by Boulch (2018).

BOULCH, A.; CHERRIER, N.; CASTAINGS, T. Ionospheric activity prediction using convolutional recurrent neural networks. arXiv:1810.13273 [cs], 6 nov. 2018. 
https://github.com/aboulch/tec_prediction/

## download 3 years of data (2018-2020)
python3 downloader.py 
Or download from: https://drive.google.com/file/d/1WJmLn_PruCiMqTQ0oPTtE2k_pZUgIEkj/view?usp=sharing

## create numpy representation for the data downloaded
python3 ionex_samples.py 

## Experiment configuration

The configuration file "config.csv" is used to setup many hyperparameters for each experiment, such as chosen model, input window, prediction window, train and test datasets, among others.

## Batch processing

The columns "batch_train" and "batch_test" on "config.csv" can be used to perform batch testing. Set them as True on the line that describes the experiment and run 

python3 batch_run.py 

The results will be created on the "output" folder, under a subfolder with the experiment's name.

## Training the network
python3 train.py

The "parameters.py" file is created during training. If you retrain the network, please remove it.

## Evaluating the trained network (test)
python3 evaluate.py

## Plot results
python3 plotresults.py

## Google Colab
https://colab.research.google.com/drive/1suy7ssFTbAiJpA_lcjJuEmQsiLy7nRS3?usp=sharing
