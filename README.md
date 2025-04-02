# WILL ETP - Application d'Apprentissage Automatique

WILL ETP est une application web Django permettant de charger des données, effectuer du prétraitement et entraîner des modèles d'apprentissage automatique.

## Fonctionnalités

- Chargement de fichiers CSV
- Analyse exploratoire des données
- Prétraitement des données :
  - Gestion des valeurs manquantes
  - Encodage des variables catégorielles
  - Normalisation des variables numériques
  - Sélection des features
- Entraînement de modèles :
  - Classification : Random Forest, Logistic Regression, SVM, XGBoost, KNN
  - Régression : Random Forest, Linear Regression, SVM, XGBoost, KNN
- Visualisation des résultats
- Historique des modèles

## Installation

1. Cloner le repository :
```bash
git clone https://github.com/votre-username/will-etp.git
cd will-etp
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Appliquer les migrations :
```bash
python manage.py migrate
```

5. Créer un superuser :
```bash
python manage.py createsuperuser
```

6. Lancer le serveur :
```bash
python manage.py runserver
```

7. Accéder à l'application :
```
http://localhost:8000
```

## Utilisation

1. **Chargement des données**
   - Accédez à la page "Charger des données"
   - Sélectionnez votre fichier CSV
   - Spécifiez les colonnes features et target
   - Ajoutez une description optionnelle

2. **Prétraitement**
   - Analysez les données
   - Gérez les valeurs manquantes
   - Encodez les variables catégorielles
   - Normalisez les variables numériques
   - Sélectionnez les features à utiliser

3. **Entraînement**
   - Choisissez l'algorithme approprié
   - Ajoutez des notes optionnelles
   - Lancez l'entraînement

4. **Résultats**
   - Consultez les métriques de performance
   - Visualisez les résultats
   - Comparez les modèles

## Structure du projet

```
will-etp/
├── manage.py
├── requirements.txt
├── README.md
├── server/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── versionning/
    ├── __init__.py
    ├── admin.py
    ├── apps.py
    ├── forms.py
    ├── models.py
    ├── urls.py
    ├── views.py
    ├── templatetags/
    │   ├── __init__.py
    │   └── custom_filters.py
    └── templates/
        ├── base.html
        ├── dashboard.html
        ├── upload_csv.html
        ├── preprocess_data.html
        ├── train_model.html
        ├── model_results.html
        ├── model_detail.html
        └── model_history.html
```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
