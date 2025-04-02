from django.test import TestCase, override_settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.exceptions import ValidationError
from django.conf import settings
import pandas as pd
import numpy as np
import json
from .models import CSV, TrainedModel
import io
import os
import shutil
import tempfile

# Définir le chemin du répertoire media pour les tests
TEST_MEDIA_ROOT = os.path.join(tempfile.gettempdir(), 'test_media')

@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT)
class CSVModelTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Créer le répertoire racine des tests
        if os.path.exists(TEST_MEDIA_ROOT):
            shutil.rmtree(TEST_MEDIA_ROOT)
        os.makedirs(TEST_MEDIA_ROOT)

    def setUp(self):
        # Créer le sous-répertoire pour les fichiers CSV
        csv_dir = os.path.join(TEST_MEDIA_ROOT, 'csv_files')
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        # Créer un fichier CSV de test
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        csv_file = io.StringIO()
        df.to_csv(csv_file, index=False)
        csv_content = csv_file.getvalue()
        
        self.csv_file = SimpleUploadedFile(
            "test.csv",
            csv_content.encode('utf-8'),
            content_type="text/csv"
        )

    def tearDown(self):
        # Nettoyer le répertoire de test après chaque test
        if os.path.exists(TEST_MEDIA_ROOT):
            shutil.rmtree(TEST_MEDIA_ROOT)

    def test_csv_creation(self):
        """Test la création d'un fichier CSV"""
        csv_model = CSV(
            file=self.csv_file,
            target='target'
        )
        csv_model.clean()  # Valide et extrait les informations
        csv_model.save()
        
        self.assertEqual(csv_model.filename, 'test.csv')
        self.assertEqual(csv_model.row_count, 5)
        self.assertEqual(csv_model.column_count, 3)
        self.assertTrue(csv_model.is_classification)

    def test_get_features_list(self):
        """Test la méthode get_features_list"""
        csv_model = CSV(
            file=self.csv_file,
            target='target',
            features='feature1,feature2'
        )
        csv_model.save()
        
        features = csv_model.get_features_list()
        self.assertEqual(features, ['feature1', 'feature2'])

    def test_invalid_target(self):
        """Test la validation avec une colonne cible invalide"""
        csv_model = CSV(
            file=self.csv_file,
            target='invalid_target'
        )
        
        with self.assertRaises(ValidationError):
            csv_model.clean()

    def test_get_file_info(self):
        """Test la méthode get_file_info"""
        csv_model = CSV(
            file=self.csv_file,
            target='target',
            features='feature1,feature2'
        )
        csv_model.clean()
        csv_model.save()
        
        info = csv_model.get_file_info()
        self.assertEqual(info['rows'], 5)
        self.assertEqual(info['columns'], 3)
        self.assertEqual(info['target'], 'target')
        self.assertEqual(info['features'], ['feature1', 'feature2'])

@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT)
class TrainedModelTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Créer le répertoire racine des tests
        if os.path.exists(TEST_MEDIA_ROOT):
            shutil.rmtree(TEST_MEDIA_ROOT)
        os.makedirs(TEST_MEDIA_ROOT)

    def setUp(self):
        # Créer le sous-répertoire pour les fichiers CSV
        csv_dir = os.path.join(TEST_MEDIA_ROOT, 'csv_files')
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        # Créer un fichier CSV de test
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        csv_file = io.StringIO()
        df.to_csv(csv_file, index=False)
        csv_content = csv_file.getvalue()
        
        self.csv_file = SimpleUploadedFile(
            "test.csv",
            csv_content.encode('utf-8'),
            content_type="text/csv"
        )
        
        self.csv_model = CSV(
            file=self.csv_file,
            target='target'
        )
        self.csv_model.clean()
        self.csv_model.save()

    def tearDown(self):
        # Nettoyer le répertoire de test après chaque test
        if os.path.exists(TEST_MEDIA_ROOT):
            shutil.rmtree(TEST_MEDIA_ROOT)

    def test_trained_model_creation(self):
        """Test la création d'un modèle entraîné"""
        model = TrainedModel(
            csv_file=self.csv_model,
            algorithm="Random Forest",
            accuracy=0.85,
            f1_score=0.83,
            precision=0.80,
            recall=0.86,
            training_time=1.5,
            model_params={"n_estimators": 100}
        )
        model.save()
        
        self.assertEqual(model.algorithm, "Random Forest")
        self.assertEqual(model.accuracy, 0.85)
        self.assertEqual(model.csv_file, self.csv_model)

    def test_trained_model_str_representation(self):
        """Test la représentation string du modèle entraîné"""
        model = TrainedModel(
            csv_file=self.csv_model,
            algorithm="Random Forest"
        )
        model.save()
        
        expected_str = f"Random Forest - {self.csv_model.filename}"
        self.assertEqual(str(model), expected_str)
