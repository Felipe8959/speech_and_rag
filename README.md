# Transcrição de Áudio e Respostas Automáticas com IA

Este projeto implementa uma aplicação de captura de áudio, transcrição, recuperação de informações usando um modelo RAG (Recuperação de Informação) e geração de respostas automáticas utilizando a API GPT-3.5.

## Funcionalidades

- Captura de áudio do sistema ou de microfones.
- Transcrição de áudio utilizando o Google Speech Recognition.
- Recuperação de manifestações similares em uma base de dados (usando TF-IDF e similaridade de cosseno).
- Geração de respostas automáticas com base em manifestações anteriores e a API GPT-3.5.
- Interface gráfica para interação com o sistema (Tkinter).

## Requisitos

### Bibliotecas Necessárias

Você pode instalar todas as bibliotecas necessárias executando o comando abaixo:

```bash
pip install -r requirements.txt
