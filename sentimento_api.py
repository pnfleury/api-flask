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

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Carrega o modelo
PIPELINE_PATH = "modelo_sentimento.joblib"
pipeline = None

try:
    pipeline = joblib.load(PIPELINE_PATH)
    print(f"Pipeline ML carregado com sucesso de: {PIPELINE_PATH}")
except Exception as e:
    print(f"ERRO FATAL: Falha ao carregar o pipeline: {e}")
    sys.exit(1)

# --- FUNÇÃO DE NÚCLEO (OTIMIZAÇÃO) ---
def executar_predicao(lista_textos):
    # Processamento em lote (Batch)
    predicoes = pipeline.predict(lista_textos)
    probabilidades_todas = pipeline.predict_proba(lista_textos)
    
    # Extração de componentes do pipeline (TF-IDF + Regressão Logística)
    vect = pipeline.steps[0][1]
    clf = pipeline.steps[1][1]
    
    # Coeficientes reais da Regressão Logística
    weights = clf.coef_[0] 
    
    resultados = []
    # O analyzer do Scikit-Learn já aplica o pré-processamento original do treino
    analyzer = vect.build_analyzer()

    for i, texto in enumerate(lista_textos):
        print(f"Texto: {texto[:40]}... | Predição: {predicoes[i]} | Probas: {probabilidades_todas[i]}")
        pred_classe = predicoes[i]
        proba_valor = probabilidades_todas[i].max()
        
        # EXPLICABILIDADE PURA
        # Analisa quais palavras presentes no texto têm maior peso no modelo
        importance = []
        tokens = set(analyzer(texto)) 
        
        for word in tokens:
            if word in vect.vocabulary_:
                idx = vect.vocabulary_[word]
                importance.append((word, weights[idx]))
        
        # Ordenação baseada no peso matemático (coef_)
        # Se positivo: maiores pesos. Se negativo: menores pesos.
        reverse_sort = (pred_classe == 'positivo')
        importance.sort(key=lambda x: x[1], reverse=reverse_sort)
        
        # Pega as 3 palavras mais influentes para aquela decisão
        top_features = [p[0] for p in importance[:3]]

        resultados.append({
            "comentario": texto,
            "sentimento": str(pred_classe),
            "probabilidade": round(float(proba_valor), 2),
            "topFeatures": top_features
        })
    
    return resultados

# --- ENDPOINTS ---

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json(silent=True)
    if not data or 'comentario' not in data:
        return jsonify({"error": "Campo 'comentario' obrigatório"}), 422
    
    # Usa a função otimizada passando uma lista de um único elemento
    resultado = executar_predicao([data['comentario']])
    return jsonify(resultado[0]), 200

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json(silent=True)
    if not data or 'textos' not in data:
        return jsonify({"error": "Campo 'textos' (lista) é obrigatório"}), 400
    
    lista_textos = [t.strip() for t in data['textos'] if t.strip()]
    if not lista_textos:
        return jsonify({"error": "Lista de textos vazia"}), 400

    resultados = executar_predicao(lista_textos)
    return jsonify(resultados), 200

# Tratamento de erros global
@app.errorhandler(Exception)
def handle_unexpected_error(e):
    if isinstance(e, HTTPException):
        return jsonify({"success": False, "error": e.name, "message": e.description}), e.code
    return jsonify({"success": False, "error": "Internal Error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)