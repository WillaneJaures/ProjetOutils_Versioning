from django.db import models
import numpy as np
import os
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.core.validators import FileExtensionValidator
from django.core.exceptions import ValidationError
import pandas as pd
import io


class CSV(models.Model):
    file = models.FileField(upload_to='csv_files/', validators=[FileExtensionValidator(allowed_extensions=['csv'])])
    filename = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    features = models.TextField(help_text="Liste des colonnes séparées par des virgules", blank=True, null=True)
    target = models.CharField(max_length=100)
    row_count = models.IntegerField(null=True, blank=True)
    column_count = models.IntegerField(null=True, blank=True)
    missing_values = models.JSONField(null=True, blank=True)
    data_types = models.JSONField(null=True, blank=True)
    unique_values = models.JSONField(null=True, blank=True)
    numerical_columns = models.JSONField(null=True, blank=True)
    categorical_columns = models.JSONField(null=True, blank=True)
    statistics = models.JSONField(null=True, blank=True)
    preprocessing_steps = models.JSONField(null=True, blank=True)
    is_preprocessed = models.BooleanField(default=False)
    preprocessed_file = models.FileField(upload_to='preprocessed_files/', null=True, blank=True)
    is_classification = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.filename} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"

    def get_features_list(self):
        """Retourne la liste des features sous forme de liste"""
        if self.features:
            return [f.strip() for f in self.features.split(',')]
        return []

    def get_file_info(self):
        """Retourne les informations du fichier sous forme de dictionnaire"""
        return {
            'name': self.filename,
            'rows': self.row_count,
            'columns': self.column_count,
            'target': self.target,
            'features': self.get_features_list(),
            'missing_values': sum(self.missing_values.values()) if self.missing_values else 0,
            'memory_usage': None,  # Sera calculé lors de la lecture du fichier
            'is_preprocessed': self.is_preprocessed,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M'),
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M')
        }

    def get_column_stats(self):
        """Retourne les statistiques des colonnes"""
        if not self.statistics:
            return {}
        
        stats = {}
        for col in self.get_features_list() + [self.target]:
            col_stats = {
                'type': self.data_types.get(col, 'unknown'),
                'missing': self.missing_values.get(col, 0),
                'unique': self.unique_values.get(col, 0)
            }
            
            if col in self.numerical_columns:
                col_stats.update({
                    'mean': f"{self.statistics['numerical'].get(col, {}).get('mean', 0):.2f}",
                    'std': f"{self.statistics['numerical'].get(col, {}).get('std', 0):.2f}",
                    'min': f"{self.statistics['numerical'].get(col, {}).get('min', 0):.2f}",
                    'max': f"{self.statistics['numerical'].get(col, {}).get('max', 0):.2f}"
                })
            stats[col] = col_stats
        
        return stats

    def save(self, *args, **kwargs):
        if not self.filename and self.file:
            self.filename = os.path.basename(self.file.name)
        super().save(*args, **kwargs)

    def clean(self):
        """Valide le fichier CSV et extrait les informations nécessaires"""
        try:
            # Lire directement depuis l'objet file
            if hasattr(self.file, 'temporary_file_path'):
                # Pour les grands fichiers qui sont stockés temporairement sur le disque
                df = pd.read_csv(self.file.temporary_file_path())
            else:
                # Pour les fichiers en mémoire
                self.file.seek(0)
                df = pd.read_csv(io.StringIO(self.file.read().decode('utf-8')))
            
            # Vérifier si le fichier est vide
            if df.empty:
                raise ValidationError("Le fichier CSV est vide")
            
            # Vérifier si la target existe
            if self.target not in df.columns:
                raise ValidationError(f"La colonne cible '{self.target}' n'existe pas dans le fichier. Colonnes disponibles : {', '.join(df.columns)}")
            
            # Identifier les types de colonnes
            self.numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Analyser les valeurs manquantes
            self.missing_values = df.isnull().sum().to_dict()
            
            # Déterminer les types de données
            self.data_types = df.dtypes.astype(str).to_dict()
            
            # Calculer les statistiques uniques
            self.unique_values = {col: df[col].nunique() for col in df.columns}
            
            # Déterminer si c'est un problème de classification ou régression
            target_values = df[self.target]
            
            # Si la colonne cible est numérique et a plus de 10 valeurs uniques, c'est probablement de la régression
            if pd.api.types.is_numeric_dtype(target_values) and len(target_values.unique()) > 10:
                self.is_classification = False
            else:
                self.is_classification = True
            
            # Sauvegarder les statistiques de base
            self.row_count = df.shape[0]
            self.column_count = df.shape[1]
            
            # Sauvegarder les statistiques détaillées
            numerical_stats = {}
            for col in self.numerical_columns:
                numerical_stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
            
            self.statistics = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'missing_values': self.missing_values,
                'data_types': self.data_types,
                'unique_values': self.unique_values,
                'numerical': numerical_stats
            }
            
            # Rembobiner le fichier pour une utilisation ultérieure
            self.file.seek(0)
            
        except pd.errors.EmptyDataError:
            raise ValidationError("Le fichier CSV est vide")
        except pd.errors.ParserError:
            raise ValidationError("Le fichier n'est pas un CSV valide")
        except UnicodeDecodeError:
            raise ValidationError("Le fichier CSV a un encodage non supporté. Veuillez utiliser UTF-8.")
        except Exception as e:
            raise ValidationError(f"Erreur lors de la validation du fichier : {str(e)}")
        
        return self

    class Meta:
        verbose_name = "Dataset CSV"
        verbose_name_plural = "Datasets CSV"
        ordering = ['-created_at']

@receiver(post_delete, sender=CSV)
def delete_file_on_delete(sender, instance, **kwargs):
    if instance.file:
        try:
            if os.path.isfile(instance.file.path):
                os.remove(instance.file.path)
        except Exception as e:
            print(f"Erreur lors de la suppression du fichier: {e}")

class TrainedModel(models.Model):
    ALGORITHM_CHOICES = [
        ("Random Forest", "Random Forest"),
        ("Linear Regression", "Régression Linéaire"),
        ("Logistic Regression", "Régression Logistique"),
        ("SVM", "Support Vector Machine"),
        ("XGBoost", "XGBoost"),
        ("KNN", "K-Nearest Neighbors"),
    ]

    csv_file = models.ForeignKey(CSV, on_delete=models.CASCADE, related_name='trained_models')
    algorithm = models.CharField(max_length=50, choices=ALGORITHM_CHOICES)
    accuracy = models.FloatField(null=True, blank=True)
    mse = models.FloatField(null=True, blank=True)
    r2_score = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    training_time = models.FloatField(null=True, blank=True)
    model_params = models.JSONField(null=True, blank=True)
    feature_importance = models.JSONField(null=True, blank=True)
    notes = models.TextField(blank=True, null=True)
    
    # Nouveaux champs pour les graphiques
    predictions = models.JSONField(null=True, blank=True)  # Pour stocker y_test et y_pred
    residuals = models.JSONField(null=True, blank=True)    # Pour stocker les résidus

    def __str__(self):
        return f"{self.algorithm} - {self.csv_file.filename}"

    def save(self, *args, **kwargs):
        # Ajouter des logs pour le débogage
        print(f"Sauvegarde du modèle : {self.algorithm}")
        print(f"Type de problème : {'Classification' if self.csv_file.is_classification else 'Régression'}")
        print(f"Paramètres du modèle : {self.model_params}")
        super().save(*args, **kwargs)

    class Meta:
        verbose_name = "Modèle entraîné"
        verbose_name_plural = "Modèles entraînés"
        ordering = ['-created_at']
