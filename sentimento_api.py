# Importar as ferramentas necessarias
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from logging.handlers import RotatingFileHandler
import joblib
import logging
import sys

# Configuração de logs o terminal e arquivo
# Configuração de Rodízio (Substitui o FileHandler comum)
# maxBytes=1000000 (1MB) | backupCount=5 (guarda os últimos 5 arquivos)
handler = RotatingFileHandler("analise_sentimentos.log", maxBytes=1000000, backupCount=5, encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%d-%m-%Y %H:%M:%S'))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%d-%m-%Y %H:%M:%S',
    handlers=[
        logging.FileHandler("analise_sentimentos.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicialização do app 
app = Flask(__name__)

# Carrega o modelo de treinamento
PIPELINE_PATH = (r"modelo_sentimento.joblib")
pipeline = None # Inicializa a variável do pipeline

try:
    # O pipeline é carregado uma única vez ao iniciar o servidor
    pipeline = joblib.load(PIPELINE_PATH) 
    print(f"Pipeline ML carregado com sucesso de: {PIPELINE_PATH}")
except Exception as e:
    # Se o carregamento falhar (arquivo não encontrado, corrompido, etc.), 
    # o erro é impresso e o programa encerra.
    print(f"ERRO FATAL: Falha ao carregar o pipeline: {e}")
    sys.exit(1)

# Tratamento de erros
@app.errorhandler(Exception)
def handle_unexpected_error(e):
    #Captura qualquer erro não tratado e retorna um JSON padronizado.
    # Se for um erro de rota (404) ou método não permitido (405)
    if isinstance(e, HTTPException):
        return jsonify({
            "success": False,
            "error": e.name,
            "message": e.description
        }), e.code

    # Erros internos de lógica ou do modelo (500)
    logger.error(f"Erro interno: {str(e)}")
    return jsonify({
        "success": False,
        "error": "Internal Server Error",
        "message": "Ocorreu um erro inesperado no processamento do sentimento."
    }), 500

# Rotas da API/Endpoint 
@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    # Checagem de disponibilidade do modelo
    if pipeline is None:
        # Retorna erro 503 se o modelo não foi carregado
        return jsonify({"error": "Serviço indisponível: Modelo não carregado."}), 503

    # Validação 1: O corpo da requisição é um JSON?
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "success": False, 
            "error": "Requisição invalida", 
            "message": "O corpo da requisição deve ser um JSON válido."
        }), 400

    # Validação 2: A chave 'comentario' existe e tem conteúdo?
    comment = data.get('comentario','').strip()
    if not comment:
        return jsonify({
            "success": False, 
            "error": "Erro de validação / dados inválidos", 
            "message": "O campo 'comentario' é obrigatório e não pode estar vazio."
        }), 422

    try:
        # INFERÊNCIA COM PIPELINE
        # O pipeline.predict() e .predict_proba() aceitam o texto bruto
        # e fazem a vetorização automaticamente internamente.
        
        # Faz a predição (classe)
        prediction = pipeline.predict([comment])[0]
        
        # Faz a predição (probabilidades)
        probabilities = pipeline.predict_proba([comment])[0]  
        
        #Usamos list() para garantir compatibilidade, pois pipeline.classes_ é um ndarray
        probability_value = probabilities[list(pipeline.classes_).index(prediction)]
        
        
        STOPWORDS_PT = {"o", "a", "os", "as", "de", "do", "da", "em", "um", "uma", "que", "com", "foi", "era"}
        # 2. EXPLICABILIDADE (TOP FEATURES)
        # IMPORTANTE: Verifique os nomes 'vect' e 'clf' abaixo conforme o seu Pipeline
        try:
            # Em vez de buscar por nome ('vect'), pegamos pela posição:
            # steps[0][1] é o objeto do Vetorizador
            # steps[-1][1] é o objeto do Classificador
            vect = pipeline.steps[0][1]
            clf = pipeline.steps[-1][1]

            # Pega os nomes das palavras e os pesos
            feature_names = vect.get_feature_names_out()
        
            # Verifica se é modelo linear (coef_) ou árvore (feature_importances_)
            if hasattr(clf, "coef_"):
                weights = clf.coef_[0]
                is_linear = True
            else:
                weights = clf.feature_importances_
                is_linear = False

            # Analisador para quebrar o texto igual o modelo faz (remove pontuação, etc)
            analyzer = vect.build_analyzer()
            words_in_comment = analyzer(comment)
        
            importance = []
            for word in set(words_in_comment):
            # REFINAMENTO: 
            # 1. Ignora se a palavra for muito curta (menos de 3 letras)
            # 2. Ignora se for uma stopword comum
                if len(word) > 2 and word not in STOPWORDS_PT:
                    if word in vect.vocabulary_:
                        idx = vect.vocabulary_[word]
                        importance.append((word, weights[idx]))

            # Ordenação: 
            # Se linear e positivo -> maiores pesos
            # Se linear e negativo -> menores pesos (mais negativos)
            # Se árvore -> maiores importâncias
            if is_linear:
                reverse_sort = (prediction == 'positivo')
            else:
                reverse_sort = True
        
            importance.sort(key=lambda x: x[1], reverse=reverse_sort)
        
            # Pega as 3 principais
            top_features = [p[0] for p in importance[:3]]

        except Exception as e:
            # Se ainda assim der erro, imprimimos o erro real para depurar
            print(f"Erro detalhado na explicabilidade: {e}")
            top_features = []

        # 3. RESPOSTA (Ajustada para o seu DTO Java em camelCase)
        response = {
            "comentario": comment,
            "sentimento": str(prediction), 
            "probabilidade": round(float(probability_value), 2),
            "topFeatures": top_features  # Adicionado aqui
        }
        
        return jsonify(response), 200

    except Exception as e:
        # Erro genérico durante a predição
        print(f"Erro durante a predição: {e}")
        return jsonify({"error": "Erro interno durante a análise de sentimento"}), 500

if __name__ == '__main__':
    # Utilizar um servidor WSGI de produção (como Gunicorn) para deploy final.
    # Para testes locais:
    app.run(host='127.0.0.1', port=5000, debug=True)