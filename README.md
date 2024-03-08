# Multilayer Perceptron project of 42
42 School love to make their student code from scrach basic things to be sure that we fully understand how they work before we use them.

This project aims to recreate a Neural Network in python using only numpy. I used a class to define a DenseLayer and a class for the Neural Network itself. Both my class are fully modulable and accept different weight initializor functions, loss functions, activation function, number of neurons in a hidden layer, number of hidden layers etc. 

My MLP is then used to create a program that learn form medical data if a tumor is benin or malicious.

## Notions developed in the project
- Forward propagation
- Backpropagation
- Gradient Descent
- Activation functions
- Weigth initialisor functions
- Earlystop
- BinaryCrossEntropyLoss

The main program MLP.py work in 3 steps:
- **Step 1: split dataset**
    - Slip the dataset in two parts (training and validation).
    - Create data_train.csv and data_val.csv files.
- **Step 2: Train the neural network**
    - If a model is already in the directory, it loads it and start the training
    - Otherwise it initilizes a new Neural Network model and trains it
- **Step 3: Predict**
    - Load the model and make a prediction about a tumor. 

## Possible arguments of the MLP.py program

--steps STEPS [STEPS ...], -s STEPS [STEPS ...] Steps to run
--dataset DATASET, -ds DATASET
Path to the full dataset (training and validation)
  --valPart VALPART, -v VALPART
                        Portion of the data to use for validation
  --layer LAYER [LAYER ...], -la LAYER [LAYER ...]
                        Name of the learning saving file
  --loss LOSS, -lo LOSS
                        Loss function to use
  --learningRate LEARNINGRATE, -lr LEARNINGRATE
                        Learning Rate to use
  --batchSize BATCHSIZE, -bs BATCHSIZE
                        Size of the batchs
  --epochs EPOCHS, -e EPOCHS
                        Number of epochs
  --dataToPredict DATATOPREDICT, -dtp DATATOPREDICT
                        Path to the dataset to predict
  --resetTraining, -r   Reset the learning
  --earlyStop, -es      Early stop on
  --precisionRecall, -pr
                        Show Precision and Recall


## Parametres necessaire pour le program : 

- **Step 1 est present :**
    - *OBLIGATOIRE :* 
        - dataset
    - *FACULTATIVE:*
        - valPart

- **Step 2 est present :**
    - *OBLIGATOIRE :* 
        - Step 1 est present | data_train.csv et data_val.csv present dans le repertoire courant
        - layer avec un format ok
    - *FACULTATIVE :*
        - loss
        - learning_rate
        - batch_size
        - epochs
        - r

- **Step 3 est present :** 
    - *OBLIGATOIRE :* 
        - Step 2 | saved_model.pkl present dans le repertoire
        - Step 1 | Step 2 | dataToPredict


**params :** 
- --steps -s [int]
- --dataset -ds str
- --valPart -v float
- --layer -la [int et str]
- --resetTraining -r Present ou pas. 
- --loss -lo str
- --learningRate -lr float
- --batchSize -bs int 
- --epochs -e int 
- --dataToPredict -dtp str

# To do
- Tester le programme avec differentes architecture du reseau de neurone. 
- Tester avec differents taux d'apprentissages. 


# Possibles bonus : 
- A more complex optimization function (for example : nesterov momentum, RMSprop, Adam, ...).
- A display of multiple learning curves on the same graph (really useful to compare different models). ==> il faudrait avoir les differntes architectures. 
    - Interessant a avoir a partir du moment ou j'ai enregistrer les trucs. 
    => done
- An historic of the metric obtained during training.
    - Il faut que a chaque fois que je lance un fit, j'enregistre les parametre de la simulation (alpha, batch size etc) ainsi que Trainning Loss et Accuracy et ValLoss et Accuracy. 
    => done
- The implementation of early stopping. ==> Permet de resoudre l'overfitting
    - Toutes les 5 epochs, j'enregistre ValLoss et je compare ValLoss[epoch] et ValLoss[epoch - 5]. si ValLoss a augmenter je reviens aux valeurs W et b de la derniere fois.
    => done
- Evaluate the learning phase with multiple metrics.
    => Done Precision et recall

# Lancement du programme 

```python src/MLP.py --steps 1 2 3 --dataset data/data.csv --layer 8 8 2 softmax --epochs 200 --resetTraining --valPart 0.5```
