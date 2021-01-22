# Evolving Architectures for CNNs using the GA

This work is the final project for the course _Intelligent Systems_ that I attended during my master degree at Department of Information Engineering (Padova, Italy). See `Evolving Architectures for Convolutional Neural Networks using the Genetic Algorithm.pdf` for the paper describing the presented algorithm and the produced results. See `example.txt` for an example of the generated output.

The purpose of this project is to implement a genetic algorithm (GA) to improve the architecture of a given Convolutional Neural Network (CNN) that is used to address image classification tasks.

Designing the architecture for a Convolutional Neural Network is a cumbersome task because of the numerous parameters to configure, including activation functions, layer types, and hyperparameters. With the large number of parameters for most networks nowadays, it is intractable to find a good configuration for a given task by hand. Due to the drawbacks of existing methods and limited computational resources available to interested researchers, most of these works in CNNs are typically performed by experts which use new theoretical insights and intuitions gained from experimentation.

The proposed method is based on the genetic algorithm since it does not require a rich domain knowledge. The goal of the algorithm is to help interested researchers to optimize their CNNs and to allow them to design optimal CNN architectures without the necessity of expert knowledge or a lot of trial and error.

## 1. Dataset

The method is tested using the [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html), in which the number of classes is 10. The numbers of training and validation images are, respectively, 50000 and 10000, and the size of each image is 32x32.

Two experimental scenarios are considered:

- Default Scenario: uses the default numbers of the training and validation images

- Small-data Scenario: assumes that only 5000 images are available. In particular, a randomly sample of 4500 images is used for the training and the remaining 500 images are employed for the validation

## 2. Algorithm and Flow Chart

The proposed algorithm represents an intelligent exploitation of a random search. Although randomized, the proposed algorithm is by no means random. Instead, it exploits historical information to direct the search into the region of better performance within the search space.

Over the course of many generations, the algorithm picks out the layers of the CNN architecture. It learns through random exploration and slowly begins to exploit its findings to select higher performing models. It receives the testing accuracy as a means of comparison between architectures and ultimately selects the best architecture. The entire process goes on for many generations until a fully trained suitable CNN model is generated.

<p align="center">
  <img src="https://github.com/AlessandroSaviolo/Evolving-CNNs-using-GA/blob/master/flowchart.png" width="400">
</p>

## 3. Project Structure

- `main.py` : main function, use it to set the hyperparameters (i.e., learning rate, number of epochs). It also contains the main structure of the genetic algorithm

- `network.py` : contains the network class (i.e., invidual belonging to the population) and the transformations applied by the genetic algorithm (e.g., mutation)

- `topology.py` : contains the layer class (e.g., convolutional, pooling, dense). It also contains the block class, where each block refers to a group of sequential layers in a network

- `inout.py` : contains the input convolutional neural network that need to be optimized

- `utilities.py` : contains plot functions and common functions among the different files (e.g., load dataset)

## 4. License

Copyright (C) 2021 Alessandro Saviolo
```
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
```
