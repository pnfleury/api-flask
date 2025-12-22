import streamlit as st
import requests
from requests.auth import HTTPBasicAuth  # Importação necessária para Basic Auth

# 1. Configurações da sua API REST
API_URL = "http://localhost:8080/sentiment"
API_URL_STATS = "http://localhost:8080/stats"
USUARIO = "admin" # Coloque seu usuário aqui
SENHA = "123456"     # Coloque sua senha aqui

st.title("FeedBackNow")

texto = st.text_area("Digite o comentário, reclamação, sugestão ou elogios")

if st.button("Analisar (positivo ou negativo)"):
    if texto:
        payload = {"comentario": texto}
        
        try:
            # 2. Fazendo a chamada com o parâmetro 'auth'
            response = requests.post(API_URL, json=payload, auth=HTTPBasicAuth(USUARIO, SENHA))
            
            # 3. Tratando a resposta
            if response.status_code == 200:
                st.success("Sucesso!")
                st.json(response.json())
            elif response.status_code == 401:
                st.error("Erro 401: Usuário ou senha do Basic Auth estão incorretos.")
            else:
                st.error(f"Erro na API 8080: Status {response.status_code}")
                
        except Exception as e:
            st.error(f"Não foi possível conectar: {e}")
st.sidebar.title("Painel de Controle")

if st.sidebar.button("📊 Visualizar estatisticas dos comentarios"):
    try:
        # 1. Fazendo a chamada GET
        response = requests.get(API_URL_STATS, auth=HTTPBasicAuth(USUARIO, SENHA))
        
        if response.status_code == 200:
            stats = response.json()
            
            # 2. Exibindo de forma organizada
            st.sidebar.subheader("Estatísticas dos comentarios")
            
            # Exemplo de como exibir se o JSON tiver chaves como 'total' ou 'media'
            # Adapte as chaves abaixo para o que sua API realmente retorna
            for chave, valor in stats.items():
                st.sidebar.metric(label=chave.capitalize(), value=valor)
                
        else:
            st.sidebar.error(f"Erro {response.status_code} ao buscar stats.")
            
    except Exception as e:
        st.sidebar.error(f"Erro de conexão: {e}")