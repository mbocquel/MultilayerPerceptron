# Multilayer Perceptron project of 42

## Notes

Dans le reseau de neurones, il faut que le premier layer (l'input layer) ait un neurone par feature de mes donnees d'entree. 

et ensuite il n'applique pas de poids autre que un neurone par feature ==> On peut utiliser une matrice avec des 1 et des 0. 

et on applique la fonction d'activation. 


entre un calque et un autre, la taille de la matrice W va dependre du nombre de neurone dans le calque d'avant et du nombre de neurone dans le calque courant. 

Dans le cas Calque 1 avec 4 neurones => calque 2 avec 3 neurones.

N1_1       
N1_2        N2_1
N1_2        N2_2
N1_4        N2_3


Par exemple le resultat de ce qui sort de N2_1 est :
function_activation(W.a_in + b)
soit avec a_in de la forme [a_1, a_2, a_3, a_4]
et w = [w2_1_1, W2_1_2, W2_1_3, W2_1_4]

f_act(W2_1_1*a_1 + W2_1_2*a_2 + W_2_1_3*a_3 + W_2_1_4*a_4 + b)


