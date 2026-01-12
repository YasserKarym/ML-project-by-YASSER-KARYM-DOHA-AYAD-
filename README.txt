Sujet 2 — Classification avec SVM sur le dataset IRIS
=====================================================
    realisee par : [DOHA AYAD ET YASSIR KARYM ]
Description
-----------

Ce projet met en œuvre un classifieur SVM (Support Vector Machine) pour la
classification supervisée sur le dataset IRIS, en utilisant la bibliothèque
scikit-learn.

Le notebook "sujet2_ML.ipynb" suit un pipeline complet de Machine Learning :
exploration des données, prétraitement, définition d'une baseline, entraînement
d'un SVM avec recherche d'hyperparamètres, évaluation détaillée et discussion
des limites du modèle.

Contenu du notebook
-------------------

Le notebook "sujet2_ML.ipynb" contient les étapes suivantes :

1) Chargement et exploration des données
----------------------------------------

- Chargement du dataset IRIS avec `sklearn.datasets.load_iris`.
- Affichage des dimensions de X (features) et y (étiquettes).
- Affichage des premières lignes des données sous forme de DataFrame.
- Calcul de statistiques descriptives (moyenne, écart-type, min, max, etc.).
- Vérification de la présence de valeurs manquantes.
- Visualisation des données via un pairplot (seaborn) pour observer la
  séparation entre les trois classes (setosa, versicolor, virginica).

Objectif : comprendre la structure des données et vérifier qu’elles sont propres.

2) Prétraitement et séparation train / test
-------------------------------------------

- Séparation des données en :
  - un ensemble d'entraînement (80 %),
  - un ensemble de test (20 %),
  avec `train_test_split`, `random_state=42` et `stratify=y`.
- Le prétraitement (standardisation) est réalisé plus tard dans un `Pipeline`
  avec `StandardScaler`.

Objectif : disposer d’un jeu de train pour l’apprentissage et d’un jeu de test
indépendant pour l’évaluation finale.

3) Baseline (modèle simple)
---------------------------

- Calcul de la classe la plus fréquente dans le jeu d'entraînement.
- Construction d’une baseline qui prédit toujours cette classe majoritaire.
- Évaluation de cette baseline sur le jeu de test avec :
  - l'accuracy,
  - le F1-score macro.

Objectif : avoir un point de comparaison naïf. Le SVM doit faire clairement
mieux que cette baseline pour être intéressant.

4) Modèle SVM avec Pipeline
---------------------------

- Création d’un `Pipeline` scikit-learn composé de :
  - `StandardScaler` pour standardiser les features (moyenne 0, variance 1),
  - `SVC` (Support Vector Classifier) comme modèle de classification.
- Le SVM est configuré pour pouvoir calculer des probabilités (`probability=True`)
  afin de tracer les courbes ROC.

Objectif : intégrer le prétraitement et le modèle dans une seule chaîne
pour éviter les erreurs de fuite de données.

5) Recherche d’hyperparamètres (GridSearchCV)
---------------------------------------------

- Définition d’une grille d’hyperparamètres pour le SVM, par exemple :
  - C ∈ {0.1, 1, 10, 100}
  - kernel ∈ {linear, rbf}
  - gamma ∈ {scale, 0.01, 0.1}
- Utilisation de `GridSearchCV` avec :
  - `scoring="f1_macro"` pour optimiser le F1-score macro,
  - `cv=5` pour une validation croisée à 5 plis,
  - `n_jobs=-1` pour paralléliser les calculs.
- Affichage des meilleurs hyperparamètres trouvés et du meilleur score moyen
  de validation croisée.

Objectif : trouver automatiquement la combinaison d’hyperparamètres qui donne
le meilleur compromis de performance sur le jeu d’entraînement.

6) Évaluation sur le jeu de test
--------------------------------

- Récupération du meilleur modèle (`best_estimator_`) obtenu par GridSearchCV.
- Prédictions des étiquettes sur le jeu de test.
- Calcul de :
  - l’accuracy sur le test,
  - le F1-score macro,
  - le rapport de classification détaillé (précision, rappel, F1 par classe).

Objectif : mesurer les performances finales du SVM sur des données jamais vues
pendant l’entraînement ou la validation croisée.

7) Matrice de confusion et courbes ROC
--------------------------------------

- Calcul et affichage d’une matrice de confusion (via `confusion_matrix` et
  `seaborn.heatmap`) pour visualiser les erreurs entre les trois classes.
- Calcul et tracé de courbes ROC multi-classes en one-vs-rest :
  - binarisation des étiquettes avec `label_binarize`,
  - calcul de FPR/TPR et AUC pour chaque classe avec `roc_curve` et `auc`.

Objectif : analyser où le modèle se trompe le plus et vérifier la qualité
de séparation entre les classes.

8) Interprétation, limites et pistes d'amélioration
---------------------------------------------------

Le notebook se termine par un commentaire (Markdown) qui explique :

- Les **performances** obtenues :
  - le SVM surpasse nettement la baseline,
  - les scores d’accuracy et de F1-macro sont élevés sur IRIS.
- Les **limites** :
  - dataset petit et simple, peu de bruit,
  - SVM sensible au choix de C, kernel et gamma,
  - interprétabilité limitée avec kernel RBF.
- Les **pistes d’amélioration** :
  - tester d’autres noyaux (poly, etc.) et d’autres grilles,
  - comparer avec d’autres modèles (RandomForest, k-NN, régression logistique),
  - utiliser une validation croisée imbriquée (nested CV),
  - appliquer la même méthodologie sur des jeux de données plus complexes
    (breast_cancer, digits, etc.).

Fichiers
--------

- sujet2_ML.ipynb : notebook principal (Google Colab).
- README.txt : ce fichier de description.

Instructions d'exécution (Google Colab)
---------------------------------------

1. Ouvrir Google Colab :
   https://colab.research.google.com/

2. Importer ou ouvrir le notebook "sujet2_ML.ipynb".

3. Vérifier que le runtime est en Python 3.

4. Exécuter les cellules dans l'ordre (Runtime > Run all / Exécuter tout).

Dépendances principales
-----------------------

- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Sur Google Colab, ces bibliothèques sont déjà installées par défaut.

Jeu de données
--------------

- Dataset : IRIS
- Chargeur : sklearn.datasets.load_iris

Le dataset IRIS contient 150 échantillons de fleurs de 3 classes
(Iris setosa, Iris versicolor, Iris virginica) avec 4 variables
continues (longueur et largeur des sépales et pétales).

Références
----------

- Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems".
  Annals of Eugenics, 7(2), 179–188.

- Documentation scikit-learn (dataset IRIS) :
  https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset