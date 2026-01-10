#  📊Motor de IA Python/Flask 
* Este microsserviço é responsável pelo processamento de linguagem natural (NLP) do sistema. Ele recebe textos de feedbacks através de uma interface REST e utiliza modelos estatísticos para classificar o sentimento do usuário.v

## 1. 🚀 Tecnologias
* Python 3.10+ / Flask: Servidor leve para exposição do modelo.
* scikit-learn (1.6.1): Biblioteca de Machine Learning utilizada no treinamento e predição.
* Joblib: Carregamento do modelo serializado.

## 2. 📊 O Modelo
O modelo foi treinado para identificar padrões semânticos em textos de clientes, categorizando-os em:
* Positivo: Feedbacks de satisfação e elogios.
* Negativo: Críticas, reclamações ou insatisfações.

## 3. API Endpoints
### 3.1. Predição do sentimento:
#### POST /sentiment
* Corpo da Requisição (JSON)
  
  ```json
    {
     "comentario": "O aparelho é muito potente, porém a entrega demorou demais.",
     "threshold": 0.5
    }
  ```
Detalhamento dos Campos
* "comentario" é um tipo string obrigatório.
* "threshold" é um tipo float que pode ser opcional.  
  Observação sobre o Threshold: Este campo permite ajustar o rigor da classificação.  
  Por padrão (0.5), qualquer predição com probabilidade superior a esse valor é marcada como POSITIVO.  
  Se você deseja que o modelo seja mais criterioso para classificar algo como positivo, você pode aumentar este valor (ex: 0.8)

* Resposta (JSON)  

```json
{
  "id": 102,
  "sentimento": "NEGATIVO",
  "probabilidade": 0.73,
  "topFeatures": [
    "aparelho",
    "demais",
    "entrega"
  ],
  "criadoEm": "30/12/2025 10:15:30"
}
```
Detalhamento dos Campos  

* "sentimento" mostra o resultado da predição (positivo ou negativo).
* "probabilidade" mostra a probabilidade (confiança).
* "topFeatures" mostra as palavras de maior peso na predição.
* "criadoEm" data e hora da resposta. 

### 3.2. Predição em lote 
#### POST /batch  
* Utilizado para análises em lote.  
* Aceita arquivos via multipart/form-data (ex: arquivo .csv).

## 4. 🐳 Integração Docker
* Este serviço foi desenhado para rodar dentro de um container, sendo consumido pela API Java (Spring Boot) através da rede interna do Docker Compose.

## 5. 🔬 Pesquisa e Treinamento (Notebook)
* Todo o processo de análise exploratória, pré-processamento de texto e treinamento do modelo pode ser visualizado no Google Colab:
* [Link para o Notebook do Projeto] (Substitua pelo seu link aqui)

## 6. 🧪Execução e Testes
* No diretório raiz da API Python/Flask, execute: ```docker-compose up --build```. 
* O serviço estará disponível em http://localhost:5000.
* Use o Postman ou Insomnia para validar o funcionamento enviando uma requisição POST para:  
* http://localhost:5000/sentiment 
```json
    {
     "comentario": "Adorei o produto",
     "threshold": 0.5
    }
  ```  
  
<br>

***
🚀Desenvolvido durante o Hackathon - FeedbackNow Team 


 
