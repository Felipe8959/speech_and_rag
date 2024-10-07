# Transcrição de Áudio e Respostas Automáticas com IA

Este projeto implementa uma aplicação de captura de áudio, transcrição, recuperação de informações usando um modelo RAG (Recuperação de Informação) e geração de respostas automáticas.

## Funcionalidades

- Captura de áudio do sistema ou de microfones.
- Transcrição de áudio utilizando o Google Speech Recognition.
- Recuperação de manifestações similares em uma base de dados (usando TF-IDF e similaridade de cosseno).
- Geração de respostas automáticas com base em manifestações anteriores.
- Interface gráfica para interação com o sistema.

## Como Executar o Projeto

1. Clone o repositório para sua máquina:

```bash
git clone https://github.com/SEU_USUARIO/transcricao_audio_rag.git
cd transcricao_audio_rag
```

2. Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

3. Execute o script principal para iniciar a interface gráfica:
```bash
python speech_audio.py
```

## Configurações Opcionais

Você pode alterar o dispositivo de entrada de áudio selecionando-o na interface gráfica. O dispositivo escolhido será salvo como padrão para execuções futuras.

## Estrutura do Projeto

`rag.py`: Contém a lógica de recuperação de informações utilizando TF-IDF e geração de respostas automáticas com GPT-3.5.

`speech_audio.py`: Script principal que gerencia a captura de áudio, transcrição e interação com o usuário.

`dados_exemplo.xlsx`: Exemplo de base de dados utilizada para recuperação de manifestações e respostas.


## Exemplo de Uso

Ao iniciar a aplicação, selecione o dispositivo de áudio, capture e transcreva o áudio. O sistema automaticamente buscará manifestações similares na base de dados e gerará uma resposta automática.
