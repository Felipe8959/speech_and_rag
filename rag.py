import os
import re
import numpy as np
import pandas as pd
from unidecode import unidecode
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('stopwords')
# nltk.download('wordnet')

mensagem_cliente = ""

# Configuração API
client = OpenAI()
api = os.getenv("OPENAI_API_KEY")
client.api_key = api


# Dados
dados = pd.read_excel('data/dataset.xlsx')
df_dados = pd.DataFrame(dados)


def limpar_texto(texto):
    # Converte para string (caso haja valores NaN ou não-string)
    texto = str(texto)
    # Converte para minúsculas
    texto = texto.lower()
    # Remove acentuações
    texto = unidecode(texto)
    # Remove caracteres especiais e mantém apenas letras, números e espaços
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    return texto


# Transformando as respostas em lista
manifestacoes = df_dados['texto_manif'].tolist()


# Encontrar a manifestação mais similar
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('portuguese'))

def preprocess_text(text):
    text = limpar_texto(text)
    
    # tokenizar, lematizar e remover stopwords
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Pré-processar as manifestações antes de gerar a matriz TF-IDF
manifestacoes_processadas = [preprocess_text(m) for m in manifestacoes]

# Gerar a matriz TF-IDF com os textos já pré-processados
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # unigramas e bigramas
tfidf_matrix = vectorizer.fit_transform(manifestacoes_processadas)

mensagem_cliente_processada = preprocess_text(mensagem_cliente)


def recuperar_resposta(mensagem_cliente_text):
    # Pré-processar a mensagem do cliente para manter consistência com a base
    mensagem_cliente_text = preprocess_text(mensagem_cliente_text)

    # Vetorizar a nova manifestação do cliente
    mensagem_cliente_tfidf = vectorizer.transform([mensagem_cliente_text])

    # Calcular a similaridade de cosseno com todas as manifestações da base
    similaridades = cosine_similarity(mensagem_cliente_tfidf, tfidf_matrix)

    # Encontrar o índice da manifestação mais similar
    indice_mais_similar = similaridades.argmax()
    similaridade_maxima = similaridades[0, indice_mais_similar]

    threshold = 0.1  # Limiar ajustado

    if similaridade_maxima >= threshold:
        # Manifestação mais similar
        manifestacao_similar = manifestacoes[indice_mais_similar]
        
        # Recuperar a resposta associada com base na manifestação
        resposta_recuperada = df_dados.loc[df_dados['texto_manif'] == manifestacao_similar, 'texto_resposta'].values[0]
    else:
        # Caso não alcance o limiar, retorna uma mensagem padrão
        manifestacao_similar = None
        resposta_recuperada = "Desculpe, não conseguimos encontrar uma resposta adequada no momento."

    return resposta_recuperada, manifestacao_similar


# teste para recuperar a manifestação e a resposta mais similar da base de dados
resposta_recuperada, manifestacao_similar = recuperar_resposta(mensagem_cliente_processada)

# print(f"\nMensagem cliente: {mensagem_cliente}")
# print(f"\nMensagem recuperada: {manifestacao_similar}")
# print(f"\nResposta recuperada: {resposta_recuperada}")


# Gerar resposta final usando GPT-3.5 com a resposta recuperada
def gerar_resposta_automatica(mensagem_cliente):    
    # chamada da API para gerar a resposta final
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Você é um atendente que responde clientes de forma educada, objetiva e clara."},
            
            # contexto com a resposta recuperada
            {"role": "user", "content": f"O cliente disse: {mensagem_cliente}. Manifestação anterior: {manifestacao_similar}. Resposta anterior: {resposta_recuperada}. Por favor, responda ao cliente de forma clara e objetiva."}
        ],
        model="gpt-3.5-turbo",
        temperature=0.1,        # ajuste de criatividade
        max_tokens=150,         # limite de tokens para a resposta
        n=1                     # número de respostas a serem geradas
    )
    
    return response.choices[0].message.content.strip()  # apenas o texto limpo