# Evolving Architectures for CNNs using the GA

This project is part of a series of projects for the course _Deep Learning_ that I attended during my exchange program at National Chiao Tung University (Taiwan). See `task.pdf` for the details of the assignment. See `report.pdf` for the report containing the representation and the analysis of the produced results.

The purpose of this project is to implement a Feedforward Neural Network from scratch. The network is implemented for both regression and classification tasks.

The first model analyzed is the neural network for regression. The model architecture consists of four layers. The two hidden layers are used to obtain better performances, at the cost of more computational time. It uses the ReLU activation function for all layers except for the output layer (no activation function there). The neural network uses Gradient Descent in order to reduce the loss function which is the Mean Squared Error.

The second model analyzed is the neural network for classification. The model architecture consists of three layers. This choice is related to the complexity of the dataset. It uses the ReLU activation function for all layers except for the output layer which uses the Sigmoid. The neural network uses Gradient Descent in order to reduce the loss function which is the Cross-Entropy.

## 1. Dataset

- [Energy Efficiency (regression)](https://drive.google.com/open?id=1m28XzC0ve9VcNv1W8TlKV5q4yKnJEKwR)

- [Ionosphere (classification)](https://drive.google.com/open?id=1YqOs39iYhChHuNVq0rmbAscAQ-xr1_y_)

## 2. Project Structure

- `main.py` : main function, use it to change task ('r' or 'c') and hyperparameters (i.e., learning rate, number of epochs)

- `model.py` : contains the regression and classification neural network models

- `regression.py` : run regression using the relative model from model.py, use it to change the hyperparameters of the model (i.e., number of neurons)

- `classification.py` : run classification using the relative model from model.py, use it to change the hyperparameters of the model (i.e., number of neurons)

- `utilities.py` : contains plot functions and common functions among the different files (i.e., load dataset which is used both for regression and classification)

- `deep_classification.py` : deep classifier used to plot the distribution of latent features at different training stages. It contains also the deep model
