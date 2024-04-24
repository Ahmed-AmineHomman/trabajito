---
title: Revito
app_file: app.py
sdk: gradio
sdk_version: 4.26.0
---

# Revito

Une interface web simpliste vous aidant à réviser vos cours.

## Installation

Pour installer ``revito``, il vous suffit de cloner ce dépôt sur votre ordinateur (bouton vert en haut de la page sur
GitHub). Ensuite, naviguez à l'aide de votre invite de commande préférée dans le dossier dans lequel vous avez cloné ce
dépôt, et installez les dépendances de la solution :

```shell
python -m pip install -r requirements.txt
```

**Remarque** : l'application utilise la bibliothèque [`unstructured`](https://github.com/Unstructured-IO/unstructured)
pour charger et récupérer le contenu des documents fournis. Cette application, par nature, nécessite de nombreuses
dépendances qui ne sont potentiellement pas toutes prises en charge par l'installation de la bibliothèque. Ces dépendances sont listées dans [ce fichier](packages.txt). Si
l'application échoue à charger vos documents, installez toutes les dépendances listées dans le fichier précédent puis essayez de nouveau. Pour les utilisateurs de windows, veuillez vous référer à
la [documentation officielle](https://github.com/Unstructured-IO/unstructured#installing-the-library) pour les consignes d'installation.

Une fois que l'installation est terminée, vous pouvez lancer l'application en exécutant le script [app.py](app.py) :

```shell
python app.py
```

### Environnement virtuel

Il est recommandé d'installer les dépendances de la solution dans un environnement virtuel, afin d'éviter tout conflit
de version. Pour cela, commencez par créer l'environnement en question une fois dans le dossier contenant le code source
de la solution :

```shell
python -m virtualenv venv
```

La commande ci-dessus devrait faire apparaître un dossier nommé `venv` dans le dossier dans lequel vous êtes. Ce dossier
va contenir l'interpréteur Python avec lequel vous allez exécuter [app.py](app.py). Commencez donc par activer
l'environnement :

```shell
venv/Scripts/activate
```

Vous aurez alors activé l'environnement virtuel, et pourrez donc installer les dépendances
dans [requirements.txt](requirements.txt) puis lancer l'application.

## Utilisation

L'application se lance tout simplement en appelant le fichier [app.py](app.py) avec l'environnement Python de votre
choix :

```shell
python app.py
```

Pour une documentation des différents paramètres exposés par l'application, appelez [app.py](app.py) avec le
paramètre ``--help`` :

```shell
python app.py --help
```

### Authentification

L'application utilise des LLMs (pour *Large Language Models*) qui vont générer des questions de révisions puis évaluer
vos réponses. Actuellement, seuls les LLMs proposés par [Cohere](https://cohere.com/) sont pris en charge par
l'application. Cohere offre [une API](https://docs.cohere.com/) permettant d'utiliser ses LLMs. Cette API dispose d'une
version gratuite permettant d'utiliser, de manière limitée en fréquence, tous les LLMs proposés par l'entreprise.
Cependant, la limite d'appels à l'API imposée par la version gratuite permet amplement un usage personnel comme celui
implémenté par `revito`.

L'usage des LLMs de l'API Cohere nécessite de disposer d'une *clé d'API*. Cette clé peut se générer, une fois votre
compte Cohere créé, sur ls [dashboard Cohere](https://dashboard.cohere.com/api-keys). Une fois créé, vous disposez de
deux méthodes pour la renseigner à `revito` :

1. Fournir votre clé d'api via le paramètre ``--api-key`` au démarrage de l'application :
    ```shell
    python app.py --api-key API_TOKEN
    ```
   où ``API_TOKEN`` correspond à votre clé d'API.
2. Créer un fichier `.env` dans le même dossier que [app.py](app.py), contenant la ligne suivante :
   ```
   COHERE_API_KEY=API_TOKEN
   ```
   où ``API_TOKEN`` correspond à votre clé d'API. Ce fichier sera lu au démarrage par l'application et la clé sera ainsi
   chargée.