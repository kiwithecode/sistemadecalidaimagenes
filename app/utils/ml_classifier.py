import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

class QCClassifier:
    """Clasificador ML para imágenes QC usando métricas extraídas."""
    
    def __init__(self, model_path=None):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'ssim_global', 'deltaE_avg', 'deltaE_max', 'deltaL', 
            'sharpness', 'ocr_conf', 'text_found_ratio', 'codes_found_ratio'
        ]
        self.model_path = model_path
        self.is_trained = False
        
        if model_path and Path(model_path).exists():
            self.load_model()
    
    def extract_features(self, metrics_dict, extras_dict):
        """Extrae características numéricas de las métricas."""
        features = []
        
        # Métricas básicas
        features.append(metrics_dict.get('ssim_global', (0, 0))[0])
        features.append(metrics_dict.get('deltaE_avg', (0, 0))[0])
        features.append(metrics_dict.get('deltaE_max', (0, 0))[0])
        
        # Luminosidad (diferencia L*)
        lab_diff = metrics_dict.get('lab_diff', ([0, 0, 0], None))[0]
        features.append(abs(lab_diff[0]) if isinstance(lab_diff, list) and len(lab_diff) > 0 else 0)
        
        # Nitidez
        features.append(metrics_dict.get('sharpness', (0, 0))[0])
        
        # OCR
        features.append(metrics_dict.get('ocr_conf', (0, 0))[0])
        
        # Ratio de textos encontrados
        text_missing = extras_dict.get('text_missing', [])
        expected_texts = len(text_missing) + len([t for t in extras_dict.get('text_found', [])])
        text_ratio = 1.0 if expected_texts == 0 else (expected_texts - len(text_missing)) / expected_texts
        features.append(text_ratio)
        
        # Ratio de códigos encontrados
        codes_missing = extras_dict.get('codes_missing', [])
        codes_found = extras_dict.get('codes_found', [])
        total_codes = len(codes_missing) + len(codes_found)
        codes_ratio = 1.0 if total_codes == 0 else len(codes_found) / total_codes
        features.append(codes_ratio)
        
        return np.array(features).reshape(1, -1)
    
    def add_training_sample(self, metrics_dict, extras_dict, label):
        """Añade una muestra de entrenamiento."""
        if not hasattr(self, 'training_data'):
            self.training_data = {'features': [], 'labels': []}
        
        features = self.extract_features(metrics_dict, extras_dict)
        self.training_data['features'].append(features.flatten())
        self.training_data['labels'].append(1 if label == 'buena' else 0)
    
    def train(self, test_size=0.2):
        """Entrena el modelo con los datos recolectados."""
        if not hasattr(self, 'training_data') or len(self.training_data['features']) < 10:
            raise ValueError("Necesitas al menos 10 muestras de entrenamiento")
        
        X = np.array(self.training_data['features'])
        y = np.array(self.training_data['labels'])
        
        # Normalizar características
        X_scaled = self.scaler.fit_transform(X)
        
        # Dividir datos
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
        else:
            # Pocos datos, usar todos para entrenamiento
            X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y
        
        # Entrenar
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluar
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"[ML] Modelo entrenado con {len(X)} muestras")
        print(f"[ML] Precisión: {accuracy:.3f}")
        
        # Importancia de características
        importance = self.model.feature_importances_
        for i, (name, imp) in enumerate(zip(self.feature_names, importance)):
            print(f"[ML] {name}: {imp:.3f}")
        
        return accuracy
    
    def predict(self, metrics_dict, extras_dict):
        """Predice si una imagen es buena o mala."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        
        features = self.extract_features(metrics_dict, extras_dict)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': 'buena' if prediction == 1 else 'mala',
            'confidence': max(probability),
            'probabilities': {'buena': probability[1], 'mala': probability[0]}
        }
    
    def save_model(self, path=None):
        """Guarda el modelo entrenado."""
        if path is None:
            path = self.model_path
        if path is None:
            raise ValueError("No se especificó ruta para guardar")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        print(f"[ML] Modelo guardado en {path}")
    
    def load_model(self, path=None):
        """Carga un modelo entrenado."""
        if path is None:
            path = self.model_path
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"[ML] Modelo cargado desde {path}")

def load_training_data_from_reports(reports_dir):
    """Carga datos de entrenamiento desde reportes JSON existentes."""
    reports_path = Path(reports_dir)
    training_samples = []
    
    for json_file in reports_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for resultado in data.get('resultados', []):
                # Convertir métricas a formato esperado
                metrics_dict = {}
                for metric in resultado.get('metrics', []):
                    metrics_dict[metric['name']] = (metric['value'], metric['threshold'])
                
                # Añadir lab_diff
                lab = resultado.get('lab', {})
                if lab:
                    metrics_dict['lab_diff'] = (lab.get('lab_diff', [0, 0, 0]), None)
                
                extras_dict = {
                    'text_missing': resultado.get('text_missing', []),
                    'codes_missing': resultado.get('codes_missing', []),
                    'codes_found': resultado.get('codes_found', [])
                }
                
                training_samples.append({
                    'metrics': metrics_dict,
                    'extras': extras_dict,
                    'label': resultado.get('status', 'mala'),
                    'file': resultado.get('file', 'unknown')
                })
                
        except Exception as e:
            print(f"[ML] Error cargando {json_file}: {e}")
    
    return training_samples
