# Neural Network and Deep Learning Project

This project involves training a neural network on a dataset of reviews and corresponding ratings, and evaluating its performance.

## Files

- `hw2main.py`: Main script for training and evaluating the neural network
- `student.py`: Code for the neural network architecture and related functions

## Usage

To run the main script, use the following command:

python3 hw2main.py


This will load the dataset, preprocess the text, define and train the neural network, and evaluate its performance.

## Requirements

This project requires Python 3, as well as the following libraries:

- PyTorch
- TorchText

The required libraries can be installed using the following command:

pip install torch torchtext


## Configuration

The `student.py` file contains several functions and options that can be modified to change the behavior of the neural network. The `network` class defines the neural network architecture, while the `loss` class defines the loss function. The `trainValSplit`, `batchSize`, `epochs`, and `optimiser` variables in `student.py` control the training process. 

