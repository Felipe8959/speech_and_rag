from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import pandas as pd
import os
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# nltk.download('stopwords')
# nltk.download('wordnet')

client = OpenAI()

api = os.getenv("OPENAI_API_KEY")

client.api_key = api

# Dados
dados = pd.read_excel('dados_exemplo.xlsx')
df_dados = pd.DataFrame(dados)
df_dados

# Transformando as respostas em lista
manifestacoes = df_dados['texto_manif'].tolist()

# Função de pré-processamento
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('portuguese'))

def preprocess_text(text):
    # remover caracteres especiais e converter para minúsculas
    text = re.sub(r'\W', ' ', text).lower()

    # tokenizar e lematizar
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Aplicar pré-processamento aos textos do banco de dados
manifestacoes_processadas = [preprocess_text(m) for m in manifestacoes]

# Vetorização com TF-IDF usando textos pré-processados
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(manifestacoes_processadas)

# Visualização da vetorização
# nomes dos termos (as palavras do vocabulário)
terms = vectorizer.get_feature_names_out()

# converte a matriz TF-IDF para um array denso
dense_matrix = tfidf_matrix.todense()

# converte para df
df_tfidf = pd.DataFrame(dense_matrix, columns=terms)

df_tfidf

mensagem_cliente = ""

# Função para encontrar a manifestação mais similar
# Com Scikit-learn
def recuperar_resposta(mensagem_cliente):
    # Aplicar pré-processamento internamente
    mensagem_cliente_processada = preprocess_text(mensagem_cliente)
    
    if not mensagem_cliente_processada:
        print("A mensagem do cliente está vazia após o pré-processamento.")
        return "Nenhuma manifestação similar encontrada", ""

    mensagem_cliente_tfidf = vectorizer.transform([mensagem_cliente_processada])
    similaridades = cosine_similarity(mensagem_cliente_tfidf, tfidf_matrix)
    
    indice_mais_similar = similaridades.argmax()
    similaridade_maxima = similaridades.max()

    threshold = 0.2

    if similaridade_maxima >= threshold:
        manifestacao_similar = manifestacoes[indice_mais_similar]
        resposta_recuperada = df_dados.loc[df_dados['texto_manif'] == manifestacao_similar, 'texto_resposta'].values[0]
        return resposta_recuperada, manifestacao_similar
    else:
        return "Nenhuma manifestação similar encontrada", ""

# Função para gerar resposta final usando GPT-3.5 com a resposta recuperada
def gerar_resposta_automatica(mensagem_cliente, resposta_recuperada, manifestacao_similar):    
    # Verifica se a mensagem do cliente está vazia
    if not mensagem_cliente:
        return "Erro: Não há transcrição disponível para gerar uma resposta."
    
    if not resposta_recuperada or not manifestacao_similar:
        return "Erro: Nenhuma manifestação similar foi recuperada."
    
    # chamada da API para gerar a resposta final
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Você é um atendente que responde clientes de forma educada, objetiva e clara."},
            
            # contexto com a resposta recuperada
            {"role": "user", "content": f"O cliente disse: {mensagem_cliente}. Manifestação anterior: {manifestacao_similar}. Resposta anterior: {resposta_recuperada}. Por favor, retorne apenas uma resposta ao cliente de forma clara e objetiva."}
        ],
        model="gpt-3.5-turbo",
        temperature=0.1,        # ajuste de criatividade
        max_tokens=150,         # limite de tokens para a resposta
        n=1                     # número de respostas a serem geradas
    )
    
    return response.choices[0].message.content.strip()  # apenas o texto limpo
