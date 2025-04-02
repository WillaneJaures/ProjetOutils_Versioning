import matplotlib
matplotlib.use('Agg')  # Configure matplotlib to use the 'Agg' backend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import time
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Avg
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb
from .models import *
from .forms import *
import os
import seaborn as sns
from datetime import datetime
from django.core.paginator import Paginator
from django.db.models import Q
from sklearn.impute import SimpleImputer
from django.core.exceptions import ValidationError
from django.conf import settings
from django.http import HttpResponse
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import csv

def get_plot_as_base64(plt):
    """Generate a base64 encoded string of a matplotlib plot"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return image_base64

@login_required
def dashboard(request):
    # Récupérer le dernier fichier CSV
    latest_csv = CSV.objects.order_by('-created_at').first()
    
    # Récupérer les statistiques
    total_files = CSV.objects.count()
    preprocessed_files = CSV.objects.filter(is_preprocessed=True).count()
    total_models = TrainedModel.objects.count()
    
    # Récupérer les derniers modèles entraînés
    recent_models = TrainedModel.objects.order_by('-created_at')[:5]
    
    # Récupérer tous les fichiers CSV
    csv_files = CSV.objects.order_by('-created_at')
    
    context = {
        'latest_csv': latest_csv,
        'total_files': total_files,
        'preprocessed_files': preprocessed_files,
        'total_models': total_models,
        'recent_models': recent_models,
        'csv_files': csv_files
    }
    
    return render(request, 'dashboard.html', context)

@login_required
def upload_csv(request):
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Vérifier si un fichier a été uploadé
                if 'file' not in request.FILES:
                    messages.error(request, 'Aucun fichier n\'a été uploadé')
                    return render(request, 'upload_csv.html', {'form': form})
                
                # Récupérer le fichier uploadé
                uploaded_file = request.FILES['file']
                
                # Vérifier l'extension
                if not uploaded_file.name.endswith('.csv'):
                    messages.error(request, 'Le fichier doit être au format CSV')
                    return render(request, 'upload_csv.html', {'form': form})
                
                # Vérifier si le fichier est vide
                if uploaded_file.size == 0:
                    messages.error(request, 'Le fichier CSV est vide')
                    return render(request, 'upload_csv.html', {'form': form})

                # Créer les dossiers nécessaires
                os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
                os.makedirs(os.path.join(settings.MEDIA_ROOT, 'csv_files'), exist_ok=True)
                os.makedirs(os.path.join(settings.MEDIA_ROOT, 'preprocessed_files'), exist_ok=True)

                try:
                    # Lire et valider le fichier CSV
                    df = pd.read_csv(uploaded_file)
                    
                    # Vérifier si le fichier est vide
                    if df.empty:
                        messages.error(request, 'Le fichier CSV est vide')
                        return render(request, 'upload_csv.html', {'form': form})
                    
                    # Vérifier si la colonne cible existe
                    target = form.cleaned_data['target']
                    if target not in df.columns:
                        messages.error(request, f"La colonne cible '{target}' n'existe pas dans le fichier. Colonnes disponibles : {', '.join(df.columns)}")
                        return render(request, 'upload_csv.html', {'form': form})
                    
                    # Obtenir la liste des features (toutes les colonnes sauf la cible)
                    features = [col for col in df.columns if col != target]
                    
                    # Rembobiner le fichier pour la sauvegarde
                    uploaded_file.seek(0)
                    
                except pd.errors.EmptyDataError:
                    messages.error(request, 'Le fichier CSV est vide')
                    return render(request, 'upload_csv.html', {'form': form})
                except pd.errors.ParserError:
                    messages.error(request, 'Le fichier n\'est pas un CSV valide')
                    return render(request, 'upload_csv.html', {'form': form})
                except Exception as e:
                    messages.error(request, f'Erreur lors de la lecture du fichier : {str(e)}')
                    return render(request, 'upload_csv.html', {'form': form})

                # Créer et sauvegarder l'instance
                csv_file = form.save(commit=False)
                csv_file.file = uploaded_file
                csv_file.filename = uploaded_file.name
                csv_file.features = ','.join(features)  # Définir les features automatiquement
                csv_file.save()
                
                messages.success(request, 'Le fichier CSV a été chargé avec succès')
                return redirect('versionning:dashboard')
                
            except ValidationError as e:
                messages.error(request, str(e))
            except Exception as e:
                messages.error(request, f"Une erreur est survenue lors du chargement du fichier : {str(e)}")
    else:
        form = CSVUploadForm()
    
    return render(request, 'upload_csv.html', {'form': form})

@login_required
def preprocess_data(request, csv_id):
    csv_file = get_object_or_404(CSV, id=csv_id)
    
    # Charger les données au début
    try:
        df = pd.read_csv(csv_file.file.path)
    except Exception as e:
        messages.error(request, f"Erreur lors de la lecture du fichier : {str(e)}")
        return redirect('versionning:dashboard')
    
    if request.method == 'POST':
        try:
            # Gestion des valeurs manquantes
            for col in df.columns:
                if f'missing_{col}' in request.POST:
                    method = request.POST[f'missing_{col}']
                    if method == 'drop':
                        df = df.dropna(subset=[col])
                    else:
                        imputer = SimpleImputer(strategy=method)
                        df[col] = imputer.fit_transform(df[[col]])
            
            # Encodage des variables catégorielles
            for col in csv_file.categorical_columns:
                if f'encoding_{col}' in request.POST:
                    method = request.POST[f'encoding_{col}']
                    if method == 'onehot':
                        df = pd.get_dummies(df, columns=[col], prefix=[col])
                    elif method == 'label':
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col])
            
            # Normalisation des variables numériques
            for col in csv_file.numerical_columns:
                if f'scaling_{col}' in request.POST:
                    method = request.POST[f'scaling_{col}']
                    if method == 'standard':
                        scaler = StandardScaler()
                        df[col] = scaler.fit_transform(df[[col]])
                    elif method == 'minmax':
                        scaler = MinMaxScaler()
                        df[col] = scaler.fit_transform(df[[col]])
                    elif method == 'robust':
                        scaler = RobustScaler()
                        df[col] = scaler.fit_transform(df[[col]])
            
            # Sélection des features
            selected_features = request.POST.getlist('selected_features')
            if selected_features:
                df = df[selected_features + [csv_file.target]]
            
            # Créer le dossier preprocessed_files s'il n'existe pas
            preprocessed_dir = os.path.join(settings.MEDIA_ROOT, 'preprocessed_files')
            os.makedirs(preprocessed_dir, exist_ok=True)
            
            # Sauvegarder le fichier prétraité
            preprocessed_filename = f'preprocessed_{csv_file.filename}'
            preprocessed_path = os.path.join('preprocessed_files', preprocessed_filename)
            full_preprocessed_path = os.path.join(settings.MEDIA_ROOT, preprocessed_path)
            
            # Sauvegarder le DataFrame
            df.to_csv(full_preprocessed_path, index=False)
            
            # Mettre à jour le modèle CSV
            csv_file.is_preprocessed = True
            csv_file.preprocessed_file = preprocessed_path
            csv_file.preprocessing_steps = {
                'missing_values': {col: request.POST.get(f'missing_{col}') for col in df.columns if f'missing_{col}' in request.POST},
                'encoding': {col: request.POST.get(f'encoding_{col}') for col in csv_file.categorical_columns if f'encoding_{col}' in request.POST},
                'scaling': {col: request.POST.get(f'scaling_{col}') for col in csv_file.numerical_columns if f'scaling_{col}' in request.POST},
                'selected_features': selected_features
            }
            csv_file.save()
            
            messages.success(request, 'Prétraitement effectué avec succès !')
            return redirect('versionning:train_model', csv_id=csv_file.id)
            
        except Exception as e:
            messages.error(request, f'Erreur lors du prétraitement : {str(e)}')
    
    # Générer les visualisations
    plots = {}
    
    try:
        # Distribution des variables numériques
        for col in csv_file.numerical_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=col)
            plt.title(f'Distribution de {col}')
            plots[f'distributions_{col}'] = get_plot_as_base64(plt)
            plt.close()
        
        # Matrice de corrélation
        if len(csv_file.numerical_columns) > 1:
            plt.figure(figsize=(12, 8))
            sns.heatmap(df[csv_file.numerical_columns].corr(), annot=True, cmap='coolwarm')
            plt.title('Matrice de corrélation')
            plots['correlation'] = get_plot_as_base64(plt)
            plt.close()
    except Exception as e:
        messages.warning(request, f"Erreur lors de la génération des graphiques : {str(e)}")
    
    context = {
        'csv_file': csv_file,
        'plots': plots,
        'columns': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.astype(str).to_dict()
    }
    
    return render(request, 'preprocess_data.html', context)

@login_required
def train_model(request, csv_id):
    csv_file = get_object_or_404(CSV, id=csv_id)
    
    try:
        # Charger les données prétraitées si disponibles, sinon les données originales
        if csv_file.is_preprocessed and csv_file.preprocessed_file:
            preprocessed_path = os.path.join(settings.MEDIA_ROOT, str(csv_file.preprocessed_file))
            if os.path.exists(preprocessed_path):
                df = pd.read_csv(preprocessed_path)
            else:
                raise FileNotFoundError(f"Le fichier prétraité n'existe pas : {preprocessed_path}")
        else:
            df = pd.read_csv(csv_file.file.path)
        
        # Obtenir les informations du fichier
        file_info = csv_file.get_file_info()
        file_info['memory_usage'] = f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        
        # Obtenir les statistiques des colonnes
        column_stats = csv_file.get_column_stats()
        
        if request.method == 'POST':
            form = ModelTrainingForm(request.POST, csv_file=csv_file)
            if form.is_valid():
                model = form.save(commit=False)
                model.csv_file = csv_file
                
                # Séparer les features et la target
                X = df.drop(columns=[csv_file.target])
                y = df[csv_file.target]
                
                # Diviser en train et test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Créer et entraîner le modèle
                start_time = time.time()
                
                try:
                    # Initialiser le scaler pour SVM
                    scaler = None
                    if model.algorithm == 'SVM':
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    
                    # Initialiser le modèle en fonction de l'algorithme
                    if model.algorithm == 'Random Forest':
                        model_instance = RandomForestClassifier(random_state=42) if csv_file.is_classification else RandomForestRegressor(random_state=42)
                    elif model.algorithm == 'Linear Regression':
                        model_instance = LinearRegression()
                    elif model.algorithm == 'Logistic Regression':
                        model_instance = LogisticRegression(random_state=42)
                    elif model.algorithm == 'SVM':
                        model_instance = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42) if csv_file.is_classification else SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
                    elif model.algorithm == 'XGBoost':
                        model_instance = xgb.XGBClassifier(random_state=42) if csv_file.is_classification else xgb.XGBRegressor(random_state=42)
                    elif model.algorithm == 'KNN':
                        model_instance = KNeighborsClassifier() if csv_file.is_classification else KNeighborsRegressor()
                    else:
                        raise ValueError(f"Algorithme non supporté : {model.algorithm}")
                    
                    # Entraîner le modèle
                    if model.algorithm == 'SVM':
                        model_instance.fit(X_train_scaled, y_train)
                        y_pred = model_instance.predict(X_test_scaled)
                        # Sauvegarder le scaler
                        model.model_params = {
                            **model_instance.get_params(),
                            'scaler': {
                                'mean_': scaler.mean_.tolist(),
                                'scale_': scaler.scale_.tolist()
                            }
                        }
                    else:
                        model_instance.fit(X_train, y_train)
                        y_pred = model_instance.predict(X_test)
                        model.model_params = model_instance.get_params()
                    
                    # Calculer les métriques
                    if csv_file.is_classification:
                        model.accuracy = accuracy_score(y_test, y_pred) * 100
                        model.precision = precision_score(y_test, y_pred, average='weighted')
                        model.recall = recall_score(y_test, y_pred, average='weighted')
                        model.f1_score = f1_score(y_test, y_pred, average='weighted')
                    else:
                        model.r2_score = r2_score(y_test, y_pred)
                        model.mse = mean_squared_error(y_test, y_pred)
                        model.rmse = np.sqrt(model.mse)
                        model.mae = np.mean(np.abs(y_test - y_pred))
                        
                        # Sauvegarder les données pour les graphiques
                        print("Sauvegarde des données pour les graphiques...")  # Log de débogage
                        model.predictions = {
                            'y_test': y_test.tolist(),
                            'y_pred': y_pred.tolist()
                        }
                        model.residuals = (y_test - y_pred).tolist()
                        print(f"Taille des données sauvegardées - y_test: {len(y_test)}, y_pred: {len(y_pred)}")  # Log de débogage
                    
                    # Temps d'entraînement
                    model.training_time = time.time() - start_time
                    
                    # Importance des features
                    if hasattr(model_instance, 'feature_importances_'):
                        model.feature_importance = dict(zip(X.columns, model_instance.feature_importances_))
                    elif hasattr(model_instance, 'coef_'):
                        if len(model_instance.coef_.shape) == 1:
                            model.feature_importance = dict(zip(X.columns, np.abs(model_instance.coef_)))
                        else:
                            model.feature_importance = dict(zip(X.columns, np.mean(np.abs(model_instance.coef_), axis=0)))
                    
                    print("Sauvegarde du modèle...")  # Log de débogage
                    # Sauvegarder le modèle
                    model.save()
                    print("Modèle sauvegardé avec succès!")  # Log de débogage
                    
                    messages.success(request, 'Modèle entraîné avec succès !')
                    return redirect('versionning:model_results', model_id=model.id)
                    
                except Exception as e:
                    messages.error(request, f"Erreur lors de l'entraînement du modèle : {str(e)}")
                    return redirect('versionning:train_model', csv_id=csv_id)
            else:
                messages.error(request, 'Formulaire invalide. Veuillez vérifier les champs.')
        else:
            form = ModelTrainingForm(csv_file=csv_file)
        
        return render(request, 'train_model.html', {
            'form': form,
            'csv_file': csv_file,
            'file_info': file_info,
            'column_stats': column_stats,
            'is_preprocessed': csv_file.is_preprocessed
        })
        
    except FileNotFoundError as e:
        messages.error(request, str(e))
        return redirect('versionning:dashboard')
    except Exception as e:
        messages.error(request, f"Erreur lors du chargement des données : {str(e)}")
        return redirect('versionning:dashboard')

@login_required
def model_results(request, model_id):
    model = TrainedModel.objects.get(id=model_id)
    return render(request, 'model_results.html', {
        'trained_model': model,
        'is_classification': model.csv_file.is_classification,
        'metrics': {
            'Accuracy': f"{model.accuracy:.2f}%" if model.accuracy else None,
            'Precision': f"{model.precision:.4f}" if model.precision else None,
            'Recall': f"{model.recall:.4f}" if model.recall else None,
            'F1-Score': f"{model.f1_score:.4f}" if model.f1_score else None,
            'MSE': f"{model.mse:.4f}" if model.mse else None,
            'R²_Score': f"{model.r2_score:.4f}" if model.r2_score else None
        },
        'plots': generate_plots(model)
    })

@login_required
def model_history(request):
    """Vue pour afficher l'historique des modèles"""
    # Récupérer tous les modèles triés par date de création
    models = TrainedModel.objects.all().order_by('-created_at')
    
    # Filtres
    algorithm = request.GET.get('algorithm')
    dataset = request.GET.get('dataset')
    model_type = request.GET.get('type')
    
    if algorithm:
        models = models.filter(algorithm=algorithm)
    if dataset:
        models = models.filter(csv_file_id=dataset)
    if model_type:
        models = models.filter(csv_file__is_classification=(model_type == 'classification'))
    
    # Pagination
    paginator = Paginator(models, 10)
    page = request.GET.get('page')
    models = paginator.get_page(page)
    
    # Options de filtrage
    algorithms = TrainedModel.objects.values_list('algorithm', flat=True).distinct()
    datasets = CSV.objects.all()
    
    context = {
        'models': models,
        'algorithms': algorithms,
        'datasets': datasets,
        'selected_algorithm': algorithm,
        'selected_dataset': dataset,
        'selected_type': model_type,
        'title': 'Historique des modèles'
    }
    
    return render(request, 'model_history.html', context)

@login_required
def model_detail(request, model_id):
    """Vue pour afficher les détails d'un modèle spécifique"""
    model = get_object_or_404(TrainedModel, pk=model_id)
    return render(request, 'model_detail.html', {'model': model})

def generate_plots(model):
    plots = {}
    
    try:
        print("Début de la génération des graphiques...")  # Log de débogage
        print(f"Type de problème : {'Classification' if model.csv_file.is_classification else 'Régression'}")
        
        if not model.csv_file.is_classification:
            print("Génération des graphiques de régression...")  # Log de débogage
            
            # Vérifier si les données sont disponibles
            if not model.predictions or not model.residuals:
                print("Données manquantes pour les graphiques!")  # Log de débogage
                return plots
            
            # Récupérer les données sauvegardées
            y_test = np.array(model.predictions['y_test'])
            y_pred = np.array(model.predictions['y_pred'])
            residuals = np.array(model.residuals)
            
            print(f"Données chargées - y_test: {len(y_test)}, y_pred: {len(y_pred)}")  # Log de débogage
            
            # Scatter plot des prédictions vs réalité
            print("Génération du scatter plot...")  # Log de débogage
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.xlabel('Valeurs réelles')
            plt.ylabel('Prédictions')
            plt.title('Prédictions vs Réalité')
            plt.grid(True, linestyle='--', alpha=0.7)
            plots['scatter_plot'] = get_plot_as_base64(plt)
            plt.close()
            print("Scatter plot généré avec succès!")  # Log de débogage
            
            # Distribution des erreurs
            print("Génération de l'histogramme des erreurs...")  # Log de débogage
            plt.figure(figsize=(10, 6))
            plt.hist(residuals, bins=50, edgecolor='black', color='blue', alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.8)
            plt.xlabel('Erreur de prédiction')
            plt.ylabel('Fréquence')
            plt.title('Distribution des erreurs')
            plt.grid(True, linestyle='--', alpha=0.7)
            plots['residual_hist'] = get_plot_as_base64(plt)
            plt.close()
            print("Histogramme des erreurs généré avec succès!")  # Log de débogage
        
        # Feature importance ou coefficients
        print("Génération du graphique d'importance des features...")  # Log de débogage
        if not model.feature_importance:
            print("Données d'importance des features manquantes!")  # Log de débogage
            return plots
            
        plt.figure(figsize=(10, 6))
        importance = pd.Series(model.feature_importance).sort_values(ascending=True)
        importance.plot(kind='barh', color='blue', alpha=0.7)
        plt.title('Coefficients du modèle' if model.algorithm == 'Linear Regression' else 'Importance des features')
        plt.xlabel('Importance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plots['feature_importance'] = get_plot_as_base64(plt)
        plt.close()
        print("Graphique d'importance des features généré avec succès!")  # Log de débogage
        
        print("Tous les graphiques ont été générés avec succès!")  # Log de débogage
        
    except Exception as e:
        print(f"Erreur lors de la génération des graphiques : {str(e)}")
        import traceback
        traceback.print_exc()
    
    return plots

@login_required
def export_results(request, model_id, format):
    """
    Exporte les résultats du modèle au format spécifié (CSV ou PDF)
    """
    model = get_object_or_404(TrainedModel, id=model_id)
    
    if format == 'csv':
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="model_{model_id}_results.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Métrique', 'Valeur'])
        
        if model.csv_file.is_classification:
            writer.writerow(['Accuracy', f"{model.accuracy:.3f}"])
            writer.writerow(['Precision', f"{model.precision:.3f}"])
            writer.writerow(['Recall', f"{model.recall:.3f}"])
            writer.writerow(['F1-Score', f"{model.f1_score:.3f}"])
        else:
            writer.writerow(['R² Score', f"{model.r2_score:.3f}"])
            writer.writerow(['MSE', f"{model.mse:.3f}"])
            writer.writerow(['RMSE', f"{model.rmse:.3f}"])
            writer.writerow(['MAE', f"{model.mae:.3f}"])
        
        writer.writerow([])
        writer.writerow(['Paramètres du modèle'])
        for param, value in model.model_params.items():
            writer.writerow([param, value])
            
        return response
        
    elif format == 'pdf':
        # Créer un document PDF
        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="model_{model_id}_results.pdf"'
        
        # Créer le PDF avec ReportLab
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Titre
        styles = getSampleStyleSheet()
        elements.append(Paragraph(f"Résultats du modèle {model.algorithm}", styles['Title']))
        elements.append(Spacer(1, 12))
        
        # Informations générales
        elements.append(Paragraph("Informations générales", styles['Heading1']))
        elements.append(Paragraph(f"Date de création : {model.created_at.strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
        elements.append(Paragraph(f"Type : {'Classification' if model.csv_file.is_classification else 'Régression'}", styles['Normal']))
        elements.append(Paragraph(f"Temps d'entraînement : {model.training_time:.2f} secondes", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Métriques
        elements.append(Paragraph("Métriques de performance", styles['Heading1']))
        data = []
        if model.csv_file.is_classification:
            data = [
                ['Métrique', 'Valeur'],
                ['Accuracy', f"{model.accuracy:.3f}"],
                ['Precision', f"{model.precision:.3f}"],
                ['Recall', f"{model.recall:.3f}"],
                ['F1-Score', f"{model.f1_score:.3f}"]
            ]
        else:
            data = [
                ['Métrique', 'Valeur'],
                ['R² Score', f"{model.r2_score:.3f}"],
                ['MSE', f"{model.mse:.3f}"],
                ['RMSE', f"{model.rmse:.3f}"],
                ['MAE', f"{model.mae:.3f}"]
            ]
        
        # Créer le tableau
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        
        # Paramètres du modèle
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Paramètres du modèle", styles['Heading1']))
        param_data = [['Paramètre', 'Valeur']]
        for param, value in model.model_params.items():
            param_data.append([param, str(value)])
        
        param_table = Table(param_data)
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(param_table)
        
        # Générer le PDF
        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()
        response.write(pdf)
        
        return response
    
    else:
        messages.error(request, f"Format d'export '{format}' non supporté.")
        return redirect('versionning:model_results', model_id=model_id)

@login_required
def delete_csv(request, csv_id):
    """Vue pour supprimer un fichier CSV et ses données associées"""
    csv_file = get_object_or_404(CSV, id=csv_id)
    
    try:
        # Supprimer le fichier physique
        if csv_file.file and os.path.exists(csv_file.file.path):
            os.remove(csv_file.file.path)
        
        # Supprimer le fichier prétraité s'il existe
        if csv_file.preprocessed_file and os.path.exists(os.path.join(settings.MEDIA_ROOT, str(csv_file.preprocessed_file))):
            os.remove(os.path.join(settings.MEDIA_ROOT, str(csv_file.preprocessed_file)))
        
        # Supprimer l'enregistrement de la base de données
        csv_file.delete()
        
        messages.success(request, 'Le fichier CSV et ses données associées ont été supprimés avec succès.')
    except Exception as e:
        messages.error(request, f"Erreur lors de la suppression du fichier : {str(e)}")
    
    return redirect('versionning:dashboard')
