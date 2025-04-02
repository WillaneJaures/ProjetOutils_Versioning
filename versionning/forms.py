import pandas as pd
from django import forms
from .models import CSV, TrainedModel


class CSVUploadForm(forms.ModelForm):
    class Meta:
        model = CSV
        fields = ['file', 'target']
        labels = {
            'file': 'Fichier CSV',
            'target': 'Colonne cible'
        }
        widgets = {
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv'
            }),
            'target': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Nom de la colonne à prédire'
            })
        }
        help_texts = {
            'file': 'Sélectionnez un fichier CSV contenant vos données',
            'target': 'Entrez le nom de la colonne que vous souhaitez prédire'
        }

    def clean_file(self):
        file = self.cleaned_data['file']
        if not file:
            raise forms.ValidationError("Veuillez sélectionner un fichier CSV")
        if not file.name.endswith('.csv'):
            raise forms.ValidationError("Le fichier doit être au format CSV")
        return file

    def clean_target(self):
        target = self.cleaned_data['target']
        if not target:
            raise forms.ValidationError("Veuillez spécifier la colonne cible")
        return target

    def clean(self):
        cleaned_data = super().clean()
        if 'file' in cleaned_data and 'target' in cleaned_data:
            try:
                # Vérifier que le fichier est un CSV valide
                file = cleaned_data['file']
                # Lire directement depuis l'objet UploadedFile
                df = pd.read_csv(file, encoding='utf-8')
                
                # Vérifier que la colonne cible existe
                target = cleaned_data['target']
                if target not in df.columns:
                    raise forms.ValidationError(
                        f"La colonne cible '{target}' n'existe pas dans le fichier CSV. Colonnes disponibles : {', '.join(df.columns)}"
                    )
                
                # Rembobiner le fichier pour une utilisation ultérieure
                file.seek(0)
                
            except pd.errors.EmptyDataError:
                raise forms.ValidationError("Le fichier CSV est vide")
            except pd.errors.ParserError:
                raise forms.ValidationError("Le fichier n'est pas un CSV valide")
            except UnicodeDecodeError:
                # Essayer avec un autre encodage si utf-8 échoue
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding='latin1')
                    file.seek(0)
                except Exception:
                    raise forms.ValidationError("Le fichier CSV a un encodage non supporté. Veuillez utiliser UTF-8 ou Latin-1")
            except Exception as e:
                raise forms.ValidationError(f"Erreur lors de la validation du fichier : {str(e)}")
        
        return cleaned_data


class ModelTrainingForm(forms.ModelForm):
    def __init__(self, *args, csv_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Définir les algorithmes disponibles selon le type de problème
        if csv_file:
            if csv_file.is_classification:
                self.fields['algorithm'].choices = [
                    ('Random Forest', 'Random Forest'),
                    ('Logistic Regression', 'Régression Logistique'),
                    ('SVM', 'SVM'),
                    ('XGBoost', 'XGBoost'),
                    ('KNN', 'K-Nearest Neighbors')
                ]
            else:
                self.fields['algorithm'].choices = [
                    ('Random Forest', 'Random Forest'),
                    ('Linear Regression', 'Régression Linéaire'),
                    ('SVM', 'SVR'),
                    ('XGBoost', 'XGBoost'),
                    ('KNN', 'K-Nearest Neighbors')
                ]
    
    class Meta:
        model = TrainedModel
        fields = ['algorithm']
        labels = {
            'algorithm': 'Algorithme'
        }
        widgets = {
            'algorithm': forms.Select(attrs={
                'class': 'form-control'
            })
        }
        help_texts = {
            'algorithm': 'Sélectionnez l\'algorithme à utiliser pour l\'entraînement'
        }
