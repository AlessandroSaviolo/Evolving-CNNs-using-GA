# Evolving Architectures for CNNs using the GA

This work is the final project for the course _Intelligent Systems_ that I attended during my master degree at Department of Information Engineering (Padova, Italy). See `Evolving Architectures for Convolutional Neural Networks using the Genetic Algorithm.pdf` for the paper describing the presented algorithm and the produced results. See `example.txt` for an example of the generated output.

The purpose of this project is to implement a genetic algorithm (GA) to improve the architecture of a given Convolutional Neural Network (CNN) that is used to address image classification tasks.

assemble a robot and implement the algorithm for making the robot smartly move in a maze and find a LED puck. Both Raspberry Pi and Arduino are used.

Designing the architecture for a Convolutional Neural Network is a cumbersome task because of the numerous parameters to configure, including activation functions, layer types, and hyperparameters. With the large number of parameters for most networks nowadays, it is intractable to find a good configuration for a given task by hand. Due to the drawbacks of existing methods and limited computational resources available to interested researchers, most of these works in CNNs are typically performed by experts which use new theoretical insights and intuitions gained from experimentation.

The proposed method is based on the genetic algorithm since it does not require a rich domain knowledge. The goal of the algorithm is to help interested researchers to optimize their CNNs and to allow them to design optimal CNN architectures without the necessity of expert knowledge or a lot of trial and error.

## 1. Dataset

- [Energy Efficiency (regression)](https://drive.google.com/open?id=1m28XzC0ve9VcNv1W8TlKV5q4yKnJEKwR)

- [Ionosphere (classification)](https://drive.google.com/open?id=1YqOs39iYhChHuNVq0rmbAscAQ-xr1_y_)

## 2. Algorithm and Flow Chart

The proposed algorithm represents an intelligent exploitation of a random search. Although randomized, the proposed algorithm is by no means random. Instead, it exploits historical information to direct the search into the region of better performance within the search space.
Over the course of many generations, the algorithm picks out the layers of the CNN architecture. It learns through random exploration and slowly begins to exploit its findings to select higher performing models. It receives the testing accuracy as a means of comparison between architectures and ultimately selects the best architecture. The entire process goes on for many generations until a fully trained suitable CNN model is generated.



## 2. Project Structure

- `main.py` : main function, use it to change task ('r' or 'c') and hyperparameters (i.e., learning rate, number of epochs)

- `model.py` : contains the regression and classification neural network models

- `regression.py` : run regression using the relative model from model.py, use it to change the hyperparameters of the model (i.e., number of neurons)

- `classification.py` : run classification using the relative model from model.py, use it to change the hyperparameters of the model (i.e., number of neurons)

- `utilities.py` : contains plot functions and common functions among the different files (i.e., load dataset which is used both for regression and classification)

- `deep_classification.py` : deep classifier used to plot the distribution of latent features at different training stages. It contains also the deep model
