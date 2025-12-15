import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from utils import DropFeatures, OneHotEncodingNames, OrdinalFeature, MinMaxWithFeatNames
from sklearn.pipeline import Pipeline

# Configuração da Página
st.set_page_config(page_title="Obesidade Analytics & Predictor", page_icon="🩺", layout="wide")

# Título Principal
st.title("🩺 Sistema de Predição e Análise de Obesidade")
st.markdown("---")

# Abas da Aplicação
tab1, tab2, tab3 = st.tabs(["ℹ️ Sobre", "🔮 Preditor de Risco", "📊 Dashboard Analítico"])

# --- Carregamento de Recursos ---
@st.cache_data
def load_data():
    # Carregar dados brutos
    df = pd.read_csv('Obesity.csv')
    return df

@st.cache_resource
def load_model_assets():
    model = joblib.load('modelo/xgb.joblib')
    label_encoder = joblib.load('modelo/label_encoder.joblib')
    # Carregando df_clean apenas para pegar a estrutura das colunas e uma linha de referência
    df_clean = pd.read_csv('dados/df_clean.csv')
    return model, label_encoder, df_clean

try:
    df_raw = load_data()
    model, label_encoder, df_clean_ref = load_model_assets()
    # Pega uma amostra para concatenação (truque do pipeline stateless)
    df_raw_reference = df_raw.drop('Obesity', axis=1)
except Exception as e:
    st.error(f"Erro ao carregar arquivos: {e}")
    st.stop()


# --- ABA 1: SOBRE ---
with tab1:
    st.header("Sobre o Projeto")
    st.markdown("### Projeto de Data Science - Análise e Predição de Obesidade")
    st.markdown("""
    Este projeto foi desenvolvido como parte do Tech Challenge (Fase 4) da Pós-Tech em Data Analytics.
    O objetivo é criar um sistema preditivo utilizando Machine Learning (XGBoost) para auxiliar profissionais 
    de saúde na identificação precoce de riscos de obesidade, analisando fatores como hábitos alimentares, 
    atividade física e dados demográficos.
    
    A aplicação conta com uma pipeline de processamento de dados completa, transformando variáveis categóricas 
    e normalizando dados numéricos para alimentar um modelo de alta precisão.
    """)
    
    st.markdown("### Participantes do Grupo")
    st.markdown("""
    - **Juan Cordeiro**: juan-bloc@hotmail.com
    - **Kaique Manoel Angelo de Paula Cardoso**: kaique.angello.01@gmail.com
    - **Lucas Alexandre Nunes de Melo**: lucasnunes.work@gmail.com
    - **Maiquel Roniele Machado de Oliveira**: maiquelroniele@gmail.com
    - **Robson Alessio**: robson.alessio@hotmail.com
    """)
    
    st.info("Desenvolvido com Streamlit, Pandas, Scikit-Learn e XGBoost.")

# --- ABA 2: PREDITOR ---
with tab2:
    st.markdown("### Preencha os dados do paciente")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Dados Pessoais")
        input_gender = st.selectbox('Gênero', ['Masculino', 'Feminino'])
        gender_map = {'Masculino': 'Male', 'Feminino': 'Female'}
        input_gender = gender_map[input_gender]
        
        input_age = st.number_input('Idade (anos)', 10, 100, 25)
        input_height = st.number_input('Altura (m)', 1.20, 2.50, 1.70)
        input_weight = st.number_input('Peso (kg)', 30.0, 300.0, 70.0)
        
        hist_map = {'Sim': 'yes', 'Não': 'no'}
        input_family_history_pt = st.radio('Histórico Familiar de Obesidade?', ['Sim', 'Não'])
        input_family_history = hist_map[input_family_history_pt]

    with col2:
        st.subheader("Hábitos Alimentares")
        input_favc_pt = st.radio('Ingere alimentos calóricos com frequência?', ['Sim', 'Não'])
        input_favc = hist_map[input_favc_pt]
        
        input_fcvc = st.slider('Consumo de Vegetais (Freqüência)', 1.0, 3.0, 2.0, help="1: Nunca, 2: Às vezes, 3: Sempre")
        
        input_ncp = st.slider('Número de Refeições Principais (diárias)', 1, 4, 3)
        
        caec_map = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
        input_caec_pt = st.selectbox('Come entre as refeições?', list(caec_map.keys()), index=1)
        input_caec = caec_map[input_caec_pt]
        
        input_smoke_pt = st.radio('Fumante?', ['Sim', 'Não'], index=1)
        input_smoke = hist_map[input_smoke_pt]

    with col3:
        st.subheader("Estilo de Vida")
        input_ch2o = st.slider('Consumo de Água (Litros/dia)', 1.0, 3.0, 2.0)
        
        input_scc_pt = st.radio('Monitora ingestão de calorias?', ['Sim', 'Não'], index=1)
        input_scc = hist_map[input_scc_pt]
        
        input_faf = st.slider('Atividade Física (Freqüência)', 0.0, 3.0, 1.0, help="0: Nenhuma, 1: 1-2 dias, 2: 3-4 dias, 3: >4 dias")
        
        input_tue = st.slider('Tempo em Dispositivos (Horas/dia)', 0.0, 2.0, 1.0, help="0: <3h, 1: 3-6h, 2: >6h (Escala aproximada)")
        
        calc_map = {'Não': 'no', 'Às vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
        input_calc_pt = st.selectbox('Consumo de Álcool', list(calc_map.keys()), index=1)
        input_calc = calc_map[input_calc_pt]
        
        trans_map = {
            'Transporte Público': 'Public_Transportation', 
            'Caminhada': 'Walking', 
            'Carro': 'Automobile', 
            'Moto': 'Motorbike', 
            'Bicicleta': 'Bike'
        }
        input_mtrans_pt = st.selectbox('Meio de Transporte Principal', list(trans_map.keys()))
        input_mtrans = trans_map[input_mtrans_pt]

    if st.button('Calcular Risco de Obesidade', type='primary'):
        
        novo_cliente_dict = {
            'Gender': input_gender, 'Age': input_age, 'Height': input_height, 'Weight': input_weight,
            'family_history': input_family_history, 'FAVC': input_favc, 'FCVC': input_fcvc,
            'NCP': input_ncp, 'CAEC': input_caec, 'SMOKE': input_smoke, 'CH2O': input_ch2o,
            'SCC': input_scc, 'FAF': input_faf, 'TUE': input_tue, 'CALC': input_calc, 'MTRANS': input_mtrans
        }
        cols_orig = ['Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 'FCVC', 'NCP', 
                     'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']
        novo_cliente_df = pd.DataFrame([novo_cliente_dict], columns=cols_orig)

        pipeline_manual = Pipeline([
            ('feature_dropper', DropFeatures()),
            ('OneHotEncoding', OneHotEncodingNames()),
            ('ordinal_feature', OrdinalFeature()),
            ('min_max_scaler', MinMaxWithFeatNames()),
        ])

        df_concat = pd.concat([df_raw_reference, novo_cliente_df], ignore_index=True)
        df_processed_concat = pipeline_manual.fit_transform(df_concat)
        cliente_final_processado = df_processed_concat.iloc[[-1]]

        cols_modelo = df_clean_ref.columns.tolist()
        for col in cols_modelo:
            if col not in cliente_final_processado.columns:
                cliente_final_processado[col] = 0
        
        cliente_final_processado = cliente_final_processado[cols_modelo]

        try:
            pred_idx = model.predict(cliente_final_processado)[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            proba = model.predict_proba(cliente_final_processado)[0][pred_idx]
            certeza_pct = proba * 100

            st.success("Análise Concluída!")
            
            mapa_cores = {
                'Normal_Weight': '🟢 Peso Normal',
                'Overweight_Level_I': '🟡 Sobrepeso Nível I',
                'Overweight_Level_II': '🟠 Sobrepeso Nível II',
                'Obesity_Type_I': '🔴 Obesidade Tipo I',
                'Obesity_Type_II': '🔴 Obesidade Tipo II',
                'Obesity_Type_III': '🆘 Obesidade Tipo III (Mórbida)',
                'Insufficient_Weight': '🔵 Abaixo do Peso'
            }
            resultado_pt = mapa_cores.get(pred_label, pred_label)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric(label="Classificação Prevista", value=resultado_pt)
            with col_res2:
                st.metric(label="Certeza do Modelo", value=f"{certeza_pct:.1f}%")
            
            st.caption(f"Nota: O modelo tem {certeza_pct:.1f}% de probabilidade de que esta seja a categoria correta.")
            
        except Exception as e:
            st.error(f"Erro na predição: {e}")

# --- ABA 3: DASHBOARD ANALÍTICO (BI) ---
with tab3:
    # 1. Preparação e Tradução dos Dados para o Dashboard
    df_dash = df_raw.copy()

    # Dicionários de Tradução
    col_rename = {
        'Gender': 'Gênero', 'Age': 'Idade', 'Height': 'Altura', 'Weight': 'Peso',
        'family_history': 'Histórico Familiar', 'FAVC': 'Consumo Calórico', 
        'FCVC': 'Freq. Vegetais', 'NCP': 'Refeições/Dia', 
        'CAEC': 'Comer entre Refeições', 'SMOKE': 'Fumante', 
        'CH2O': 'Consumo Água', 'SCC': 'Monitora Calorias', 
        'FAF': 'Freq. Atividade Física', 'TUE': 'Tempo Eletrônicos', 
        'CALC': 'Consumo Álcool', 'MTRANS': 'Transporte', 'Obesity': 'Nível de Obesidade'
    }
    
    val_map = {
        'Male': 'Masculino', 'Female': 'Feminino',
        'yes': 'Sim', 'no': 'Não',
        'Sometimes': 'Às vezes', 'Frequently': 'Frequentemente', 'Always': 'Sempre',
        'Public_Transportation': 'Transporte Público', 'Walking': 'Caminhada', 
        'Automobile': 'Carro', 'Motorbike': 'Moto', 'Bike': 'Bicicleta',
        'Insufficient_Weight': 'Abaixo do Peso', 'Normal_Weight': 'Peso Normal',
        'Overweight_Level_I': 'Sobrepeso I', 'Overweight_Level_II': 'Sobrepeso II',
        'Obesity_Type_I': 'Obesidade I', 'Obesity_Type_II': 'Obesidade II', 
        'Obesity_Type_III': 'Obesidade III (Mórbida)'
    }

    # Aplicar traduções
    df_dash.rename(columns=col_rename, inplace=True)
    for col in df_dash.select_dtypes(include='object').columns:
        df_dash[col] = df_dash[col].replace(val_map)

    # Ordem correta das categorias de obesidade para gráficos
    ordem_obesidade = [
        'Abaixo do Peso', 'Peso Normal', 'Sobrepeso I', 'Sobrepeso II',
        'Obesidade I', 'Obesidade II', 'Obesidade III (Mórbida)'
    ]

    # --- ÁREA DE FILTROS (Estilo BI) ---
    st.markdown("### 🔎 Explorador de Dados (Filtros)")
    with st.expander("Clique para abrir os filtros e segmentar os dados", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        with f_col1:
            sel_genero = st.multiselect("Gênero", df_dash['Gênero'].unique(), default=df_dash['Gênero'].unique())
        with f_col2:
            sel_hist = st.multiselect("Histórico Familiar", df_dash['Histórico Familiar'].unique(), default=df_dash['Histórico Familiar'].unique())
        with f_col3:
            sel_transporte = st.multiselect("Transporte", df_dash['Transporte'].unique(), default=df_dash['Transporte'].unique())
        with f_col4:
            sel_fumante = st.multiselect("Fumante", df_dash['Fumante'].unique(), default=df_dash['Fumante'].unique())

    # Filtragem do DataFrame
    df_filtered = df_dash[
        (df_dash['Gênero'].isin(sel_genero)) &
        (df_dash['Histórico Familiar'].isin(sel_hist)) &
        (df_dash['Transporte'].isin(sel_transporte)) &
        (df_dash['Fumante'].isin(sel_fumante))
    ]

    st.markdown("---")

    # --- KPIs ---
    if len(df_filtered) == 0:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
    else:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Selecionado", len(df_filtered))
        media_peso = df_filtered['Peso'].mean()
        kpi2.metric("Peso Médio", f"{media_peso:.1f} kg")
        media_idade = df_filtered['Idade'].mean()
        kpi3.metric("Idade Média", f"{media_idade:.1f} anos")
        pct_risco = df_filtered['Nível de Obesidade'].apply(lambda x: 'Obesidade' in x).mean() * 100
        kpi4.metric("% Com Obesidade", f"{pct_risco:.1f}%")

        # --- LINHA 1: Distribuição e Dispersão ---
        st.subheader("📊 Visão Geral da Distribuição")
        row1_col1, row1_col2 = st.columns([1, 2]) # Coluna 2 mais larga

        with row1_col1:
            fig_donut = px.pie(df_filtered, names='Nível de Obesidade', 
                               title='Distribuição por Categoria', 
                               hole=0.5,
                               category_orders={'Nível de Obesidade': ordem_obesidade})
            fig_donut.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5))
            st.plotly_chart(fig_donut, use_container_width=True)

        with row1_col2:
            fig_scatter = px.scatter(df_filtered, x='Peso', y='Altura', color='Nível de Obesidade',
                                     size='Idade', hover_data=['Gênero', 'Transporte'],
                                     title='Análise Multivariada: Peso x Altura x Idade (Tamanho)',
                                     category_orders={'Nível de Obesidade': ordem_obesidade})
            st.plotly_chart(fig_scatter, use_container_width=True)

        # --- LINHA 2: Análise Categórica Interativa ---
        st.markdown("---")
        st.subheader("🔬 Análise de Comportamento e Hábitos")
        
        row2_col1, row2_col2 = st.columns([1, 1])

        with row2_col1:
            # Dropdown para escolher o eixo X
            opcoes_analise = ['Freq. Vegetais', 'Comer entre Refeições', 'Consumo Água', 'Freq. Atividade Física', 'Tempo Eletrônicos', 'Consumo Álcool']
            eixo_x_selecionado = st.selectbox("Selecione um Hábito para Analisar:", opcoes_analise)
            
            fig_bar = px.histogram(df_filtered, x=eixo_x_selecionado, color='Nível de Obesidade', 
                                   barmode='group', title=f"Relação: {eixo_x_selecionado} vs Obesidade",
                                   category_orders={'Nível de Obesidade': ordem_obesidade})
            st.plotly_chart(fig_bar, use_container_width=True)

        with row2_col2:
            # Gráfico de Categorias Paralelas (Sankey Simplificado)
            st.markdown("**Fluxo de Características:**")
            st.caption("Entenda como o histórico familiar e hábitos levam à obesidade.")
            fig_par = px.parallel_categories(df_filtered, dimensions=['Histórico Familiar', 'Consumo Calórico', 'Nível de Obesidade'],
                                             color='Idade', color_continuous_scale=px.colors.sequential.Inferno,
                                             title="Fluxo: Família -> Calorias -> Obesidade")
            st.plotly_chart(fig_par, use_container_width=True)

        # --- LINHA 3: Correlação (Mapa de Calor) ---
        st.markdown("---")
        st.subheader("🔥 Mapa de Correlação (Variáveis Numéricas)")
        
        # Selecionar apenas colunas numéricas para correlação
        cols_num = ['Idade', 'Altura', 'Peso', 'Freq. Vegetais', 'Refeições/Dia', 'Consumo Água', 'Freq. Atividade Física', 'Tempo Eletrônicos']
        corr_matrix = df_filtered[cols_num].corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
                             title="O que está mais relacionado? (Correlação de Pearson)")
        st.plotly_chart(fig_corr, use_container_width=True)