---
title: Trabajito
app_file: app.py
sdk: gradio
sdk_version: 4.26.0
---

# Trabajito

Une interface web simpliste vous aidant à réviser vos cours.

## Installation

Pour installer ``trabajito``, vous devez commencer par récupérer le code source. Pour cela, vous pouvez soit cloner ce
dépôt via `git` soit télécharger depuis GitHub le code source et l'extraire dans le dossier de votre choix.

Si vous souhaitez passer par `git`, il vous faut tout d'abord installer l'outil
depuis [le site officiel](https://git-scm.com/). Une fois `git` installé, ouvrez une invite de commande et tapez la
commande suivante :

```shell
git clone https://github.com/Ahmed-AmineHomman/trabajito.git mon_dossier
```

où `mon_dossier` est le nom du dossier dans lequel vous souhaitez cloner le dépôt. Si vous ne spécifiez pas de nom de
dossier, `git` créera un dossier nommé `trabajito` dans le dossier courant.

Une fois le code source récupéré, il vous faudra installer les dépendances externes, i.e. non inclus dans Python,
nécessaires pour que l'application fonctionne. Veuillez vous référer à la
section [Dépendances Externes](#dépendances-externes) pour plus d'informations sur ces dépendances.

Une fois toutes les dépendances externes installées, vous pouvez passer à l'installation de l'application à proprement
dit. Pour cela, ouvrez une invite de commande à l'endroit où vous avez placé le code source et tapez la commande
suivante :

```shell
python -m pip install .
```

**Remarque** : il est conseillé d'installer l'application dans un environnement virtuel pour éviter tout conflit de
version. Pour cela, vous pouvez suivre les instructions données dans la
section [Environnement virtuel](#environnement-virtuel).

## Démarrer l'application

Une fois que l'installation est terminée, vous pouvez lancer l'application en exécutant le script [app.py](app.py) :

```shell
python app.py
```

L'application devrait alors se lancer et vous afficher un lien (dans le terminal) vers lequel vous pouvez vous rendre
pour accéder à l'interface web.

**Remarque** : selon le choix du LLM motorisant l'application (qui dépendra probablement de votre configuration
matérielle et/ou de votre budget), il est possible que vous nécessitiez d'obtenir une clé d'API pour pouvoir utiliser
l'application. Pour plus d'informations, veuillez vous référer à la section [Authentification](#authentification).

Pour une documentation des différents paramètres exposés par l'application, appelez [app.py](app.py) avec le
paramètre ``--help`` :

```shell
python app.py --help
```

## Annexes

### Dépendances Externes

L'application utilise la bibliothèque [`unstructured`](https://github.com/Unstructured-IO/unstructured)
pour charger et récupérer le contenu des documents fournis. Cette application, par nature, nécessite de nombreuses
dépendances qui ne sont potentiellement pas toutes prises en charge par l'installation de la bibliothèque. Ces
dépendances doivent donc être installées manuellement au préalable de l'utilisation de l'application. Vous pouvez vous
référer sur la [documentation officielle](https://docs.unstructured.io/open-source/installation/full-installation) pour
plus de détails sur cette étape.

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
2. Créer une variable d'environnement `COHERE_API_KEY` sur votre système d'exploitation et définir sa valeur à votre clé
   d'API. Une fois cette variable créée, redémarrez éventuellement l'application afin que cette dernière puisse avoir
   accès à la variable que vous venez de créer.
3. Au préalable de lancer l'application, créez la variable d'environnement `COHERE_API_KEY` directement dans le
   terminal. Cela peut se faire via la commande suivante:
   ```shell
   $env:COHERE_API_KEY="API_TOKEN"; python app.py
   ```
   si vous êtes sur Windows, ou
   ```shell
   export COHERE_API_KEY="API_TOKEN"; python app.py
   ```
   si vous êtes sur un système Unix, où `API_TOKEN` correspond à votre clé d'API.

### Unstructured