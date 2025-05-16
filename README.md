ntroduction
Le suivi de la santé des colonies d’abeilles est un enjeu crucial pour l’apiculture moderne, notamment en ce qui concerne la détection rapide de l’absence de la reine, un facteur déterminant pour la survie de la ruche. Les avancées récentes en intelligence artificielle et en Internet des objets (IoT) permettent aujourd’hui d’automatiser cette surveillance grâce à l’analyse acoustique des sons produits par les abeilles. Ce travail présente le développement et l’implémentation d’un modèle d’apprentissage profond, basé sur les réseaux de neurones convolutifs (CNN), capable de classifier les bruits de ruches afin de déterminer la présence ou l’absence de la reine. L’ensemble du processus, depuis la collecte des données jusqu’au déploiement du modèle sur des dispositifs embarqués à faible consommation, est détaillé, avec une attention particulière portée à l’optimisation pour l’embarqué et à la validation des performances du système.

Voici un sommaire court et simple adapté au contenu du document :

Introduction

Collecte et préparation des données

Architecture du modèle CNN

Entraînement et résultats

Déploiement sur microcontrôleur

Conclusion
Comment automatiser la détection de l’absence de la reine dans une ruche grâce à l’analyse acoustique, afin d’aider les apiculteurs à surveiller efficacement la santé de leurs colonies sans intervention intrusive ?
1. Introduction
À quoi sert cette partie ?
L’introduction pose le contexte et explique pourquoi le sujet est important. Elle présente le problème à résoudre et l’objectif du projet.

Explication simple :
Dans cette étude, on cherche à surveiller la santé des ruches en détectant automatiquement la présence ou l’absence de la reine grâce à l’analyse des sons. La reine est essentielle pour la colonie : si elle disparaît, la ruche peut mourir. Traditionnellement, les apiculteurs doivent ouvrir les ruches régulièrement pour vérifier la présence de la reine, ce qui est intrusif et stressant pour les abeilles.

Objectif du projet :
Créer un système automatique, basé sur l’intelligence artificielle (IA) et l’Internet des objets (IoT), qui écoute les sons de la ruche et indique rapidement si la reine est présente ou non. Cela permet d’agir vite et de préserver la colonie.

1.1. Context and Motivation
À quoi sert cette partie ?
Cette section explique le contexte général et les raisons pour lesquelles ce sujet est important aujourd’hui.

Explication simple :
Pourquoi surveiller les ruches ?
Les abeilles jouent un rôle crucial dans la pollinisation et l’agriculture. Leur santé est donc essentielle pour l’environnement et l’alimentation humaine.

Le problème de la reine :
La reine est le cœur de la ruche. Si elle meurt ou disparaît, la colonie devient désorganisée et peut rapidement dépérir.

Limites des méthodes traditionnelles :
Les apiculteurs doivent ouvrir la ruche pour vérifier la reine, ce qui prend du temps, dérange les abeilles et n’est pas toujours fiable.

Pourquoi l’acoustique ?
Les abeilles produisent des sons différents selon leur état. En analysant ces sons, on peut savoir si la reine est présente sans ouvrir la ruche.

1.2. Problem Statement
À quoi sert cette partie ?
Ici, on définit clairement le problème que le projet cherche à résoudre.

Explication simple :
Problème principal :
Comment détecter automatiquement et de façon fiable l’absence de la reine dans une ruche, sans intervention humaine directe ?

Difficultés :

Les sons de la ruche sont complexes et varient selon beaucoup de facteurs (température, activité, etc.).

Il faut un système capable de distinguer les sons “normaux” des sons qui indiquent un problème (comme l’absence de la reine).

Le système doit fonctionner sur des petits appareils (microcontrôleurs) avec peu de mémoire et de puissance.

1.3. Objectives
À quoi sert cette partie ?
Cette section liste les objectifs précis que le projet veut atteindre.

Explication simple :
Objectif principal :
Développer un modèle d’IA capable de classer les sons de la ruche pour dire si la reine est présente ou absente.

Sous-objectifs :

Collecter des données : Enregistrer les sons des ruches avec et sans reine.

Prétraiter les données : Nettoyer et transformer les sons en spectrogrammes (images des sons).

Construire et entraîner un modèle IA : Utiliser un réseau de neurones (CNN) pour apprendre à reconnaître les sons caractéristiques.

Optimiser le modèle : Le rendre assez petit et rapide pour fonctionner sur un microcontrôleur dans la ruche.

Déployer le modèle : Installer le système dans la ruche pour surveiller en temps réel et alerter l’apiculteur si besoin.
2. Methodology
(Méthodologie)

À quoi sert cette partie ?
Cette section explique comment le projet a été mené, étape par étape. Elle décrit le processus suivi pour développer le système, depuis la collecte des données jusqu’au déploiement du modèle d’intelligence artificielle.

Explication simple :
La méthodologie est divisée en quatre grandes étapes (voir Figure 6 du texte) :

Collecte des données : Installer des capteurs dans les ruches pour enregistrer les sons.

Prétraitement des données : Nettoyer et préparer les sons pour l’analyse.

Extraction des caractéristiques : Transformer les sons en spectrogrammes (images qui montrent les fréquences et leur évolution dans le temps).

Déploiement du modèle : Installer le modèle sur un appareil embarqué (microcontrôleur) pour qu’il puisse fonctionner directement dans la ruche.

Figure 6 illustre ce processus comme une chaîne de traitement, où chaque étape prépare la suivante.

2.1 Data Collection and Preprocessing
(Collecte et prétraitement des données)

À quoi sert cette partie ?
Cette sous-section décrit comment les données de base (les sons des ruches) ont été obtenues et préparées pour l’analyse.

Explication simple :
Collecte des données
Où ?
Des capteurs acoustiques ont été installés dans des ruches, sur le terrain, pour enregistrer les bruits produits par les abeilles.

Comment ?
Les sons ont été enregistrés à une fréquence de 16 kHz, par séquences de 12 secondes, sur une période de 3 mois (mars à juin).

Combien ?
Environ 300 échantillons ont été collectés, représentant à peu près une heure d’enregistrement pour chaque ruche, répartis entre périodes avec reine et sans reine.

Prétraitement des données
Nettoyage
Les sons bruts sont souvent “sales” (bruits parasites, variations de volume, etc.). On les nettoie pour ne garder que les informations utiles.

Découpage
Les longues séquences sont découpées en petits morceaux (fenêtres) pour faciliter l’analyse.

Transformation en spectrogrammes
Chaque morceau de son est transformé en une image appelée spectrogramme, qui montre comment les fréquences changent au fil du temps. C’est sur ces images que le modèle d’IA va travailler.

Pourquoi cette étape est-elle importante ?
Parce que la qualité des données d’entrée détermine la performance du modèle. Si les sons sont bien enregistrés, bien nettoyés et bien transformés, le modèle pourra apprendre plus facilement à distinguer une ruche avec ou sans reine.

Résumé visuel (schéma simplifié) :
Capteurs dans la ruche → 2. Enregistrement des sons → 3. Nettoyage et découpage → 4. Transformation en spectrogrammes

En résumé :
La méthodologie commence par une collecte rigoureuse des sons de la ruche, suivie d’un prétraitement minutieux pour transformer ces sons en données exploitables par l’intelligence artificielle. Ces étapes sont essentielles pour garantir la fiabilité et l’efficacité du système de détection
3. Feature Extraction and Model Engineering
3.1 Extraction des caractéristiques (Feature Extraction)
But :
Transformer les sons enregistrés en une forme que le modèle d’IA peut comprendre et apprendre.

Comment ?

Les sons bruts sont convertis en spectrogrammes : ce sont des images où l’axe horizontal représente le temps, l’axe vertical la fréquence, et la couleur/l’intensité indique l’énergie sonore à chaque instant et fréquence.

Ces spectrogrammes permettent de visualiser et d’analyser les différences entre une ruche avec reine et une ruche sans reine.

Pourquoi c’est important ?
Les modèles d’IA, surtout les réseaux de neurones convolutifs (CNN), sont très efficaces pour reconnaître des motifs dans des images. Les spectrogrammes rendent les différences acoustiques “visibles” pour le modèle.

3.2 Modélisation (Model Engineering)
But :
Construire un modèle d’IA capable de distinguer les deux situations (reine présente ou absente) à partir des spectrogrammes.

Détails techniques :

Architecture utilisée : Un réseau de neurones convolutif 1D (Conv1D), adapté à la structure des données audio.

Structure du modèle :

Première couche de convolution : 8 filtres, détecte des motifs simples.

Seconde couche de convolution : 16 filtres, détecte des motifs plus complexes.

Pooling : Réduction de la taille des données (max pooling), ce qui simplifie le calcul et évite le sur-apprentissage.

Couches finales : Les résultats sont aplatis (flatten) puis envoyés dans une couche dense (MLP) pour la classification finale.

Hyperparamètres principaux :

Taille des filtres : 3

Stride (décalage du filtre) : 1

Pooling : taille 2, stride 2

Exemple illustré :

Figure 9 : Montre comment un filtre de taille 3 glisse sur le spectrogramme pour extraire des motifs.

Figure 10 : Montre l’étape de pooling (réduction de taille).

Figure 11 : Montre le résultat final de la convolution et du pooling, avant la classification.

4. Model Training and Validation
4.1 Entraînement du modèle
But :
Apprendre au modèle à reconnaître les sons d’une ruche avec ou sans reine.

Comment ?

Données : 300 échantillons, divisés en 75% pour l’entraînement et 25% pour le test.

Processus :

Le modèle passe 100 fois (100 epochs) sur les données d’entraînement.

À chaque passage, il ajuste ses paramètres pour améliorer la précision.

Fonction de perte utilisée : categorical cross-entropy (mesure l’écart entre la prédiction et la réalité).

Regularisation :

Dropout : désactive aléatoirement certains neurones pendant l’entraînement pour éviter le sur-apprentissage.

Résultats :

Précision (accuracy) : 98%

Rappel (recall) : 98%

F1-score : 0,98

Exemples de figures :

Figure 12 : Distribution des prédictions pendant l’entraînement (gauche) et l’inférence (droite).

Figure 13 : Courbe d’évolution de la précision.

Figure 14 : Matrice de confusion (résume les bonnes et mauvaises classifications).

Figure 15 : Validation finale sur le jeu de test.

5. Model Deployment
5.1 Déploiement sur microcontrôleur
But :
Rendre le modèle utilisable dans la vraie vie, sur un petit appareil placé dans la ruche.

Comment ?

Conversion du modèle :

Utilisation de TensorFlow Lite pour rendre le modèle compatible avec les appareils embarqués.

Application de la quantification (réduction de la taille des poids du modèle, de float32 à int8).

Compilation :

Utilisation de l’EON Compiler pour transformer le modèle en code C++ et en bibliothèque Arduino.

Performance sur microcontrôleur :

Temps d’inférence : 51 ms pour analyser 2 secondes de son.

Mémoire utilisée : 28 Ko de RAM.

Pourquoi c’est important ?
Le modèle peut fonctionner en continu dans la ruche, sans intervention humaine, et sans consommer beaucoup d’énergie.

