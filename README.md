# Multilayer Perceptron project of 42

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