import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from utils import DropFeatures, OneHotEncodingNames, OrdinalFeature, MinMaxWithFeatNames

# 1. Carregar dados
print("Carregando dados...")
df = pd.read_csv('obesidade/Obesity.csv')

# 2. Separar Target e Features
X = df.drop('Obesity', axis=1)
y = df['Obesity']

# 3. Codificar o Target (Categorias -> Números)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Salvar o LabelEncoder para usar no App (para decodificar a predição depois)
joblib.dump(label_encoder, 'obesidade/modelo/label_encoder.joblib')

# 4. Pipeline de Pré-processamento
pipeline = Pipeline([
    ('feature_dropper', DropFeatures()), # Nenhuma feature para dropar por enquanto
    ('OneHotEncoding', OneHotEncodingNames()),
    ('ordinal_feature', OrdinalFeature()),
    ('min_max_scaler', MinMaxWithFeatNames()),
])

print("Aplicando pipeline...")
# Fit_transform nos dados brutos para gerar o df_clean (referência para o app)
X_processed = pipeline.fit_transform(X)

# Salvar df_clean.csv para o app ler a estrutura
X_processed.to_csv('obesidade/dados/df_clean.csv', index=False)

# 5. Treinamento do Modelo
print("Treinando modelo XGBoost...")
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Avaliação simples
acc = model.score(X_test, y_test)
print(f"Acurácia do modelo: {acc:.2f}")

# 6. Salvar o modelo
joblib.dump(model, 'obesidade/modelo/xgb.joblib')
print("Modelo e dados salvos com sucesso!")