# 🩺 Projeto de Predição e Análise de Obesidade

Este projeto é uma aplicação web interativa desenvolvida em **Python** utilizando **Streamlit**. Ele tem como objetivo auxiliar na identificação de riscos de obesidade através de Machine Learning e fornecer uma análise exploratória detalhada dos dados demográficos e de hábitos de vida.

## 📋 Funcionalidades

O sistema é dividido em três módulos principais:

1.  **ℹ️ Sobre**: Informações sobre o projeto e a equipe desenvolvedora.
2.  **🔮 Preditor de Risco**: Um formulário interativo onde o usuário insere dados de um paciente (idade, peso, hábitos alimentares, etc.) e recebe uma predição em tempo real sobre o nível de obesidade, baseada em um modelo **XGBoost**.
3.  **📊 Dashboard Analítico**: Uma suíte de Business Intelligence (BI) com filtros dinâmicos, gráficos interativos (Plotly), mapas de calor de correlação e análise de fluxo de dados (Sankey).

---

## 🚀 Como executar o projeto

Siga os passos abaixo para configurar o ambiente e rodar a aplicação em sua máquina.

### 1. Pré-requisitos

Certifique-se de ter o **Python (versão 3.8 ou superior)** instalado.

### 2. Clonar ou Baixar o Repositório

Navegue até a pasta do projeto via terminal:

```bash
cd obesidade
```

### 3. Configurar o Ambiente Virtual (Recomendado)

É uma boa prática criar um ambiente virtual para não conflitar com outras bibliotecas do seu sistema.

**No Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**No Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar Dependências

Com o ambiente ativo (ou não), instale as bibliotecas listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 5. Executar a Aplicação

Inicie o servidor do Streamlit com o seguinte comando:

```bash
streamlit run app.py
```

O navegador abrirá automaticamente no endereço `http://localhost:8501`.

---

## 📂 Estrutura do Projeto

- `app.py`: Arquivo principal da aplicação (Frontend e Lógica).
- `utils.py`: Classes e funções auxiliares para o pipeline de processamento de dados (OneHotEncoding, Normalização, etc.).
- `requirements.txt`: Lista de dependências do projeto.
- `dados/`: Contém os datasets brutos (`Obesity.csv`) e processados (`df_clean.csv`).
- `modelo/`: Contém os artefatos do modelo treinado (`xgb.joblib`) e encoders (`label_encoder.joblib`).
- `Notebooks/`: Notebooks Jupyter utilizados para a análise exploratória e treinamento do modelo.

---

## 🛠️ Tecnologias Utilizadas

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Manipulação de Dados:** Pandas
- **Machine Learning:** Scikit-Learn, XGBoost
- **Visualização:** Plotly, Matplotlib, Seaborn

---

## 👨‍💻 Autores

Projeto desenvolvido como parte do **Tech Challenge - Fase 4 (Data Analytics)**.

- **Juan Cordeiro**
- **Kaique Manoel Angelo de Paula Cardoso**
- **Lucas Alexandre Nunes de Melo**
- **Maiquel Roniele Machado de Oliveira**
- **Robson Alessio**
