# DeepTEC
A deep learning laboratory for Total Electron Content prediction experiments, using Tensorflow 2.\
Models can be found at ./models/. Some reusable layers are on ./models/custom_layers.py.\
The c111 and c333 models were inspired by Boulch (2018).

BOULCH, A.; CHERRIER, N.; CASTAINGS, T. Ionospheric activity prediction using convolutional recurrent neural networks. arXiv:1810.13273 [cs], 6 nov. 2018. 
https://github.com/aboulch/tec_prediction/

## download IONEX data
python3 downloader.py 
Or download from: https://drive.google.com/file/d/1Sm_PiVUIabaew_3Y7sT0NWBqu7xsdHvi/view?usp=share_link

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
https://github.com/mauriciodev/tec_forecast/blob/main/examples/tec_forecast.ipynb
