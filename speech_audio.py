import os
import json
import threading
import re
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from rag import recuperar_resposta, gerar_resposta_automatica, preprocess_text

# Função para salvar o dispositivo padrão
def salvar_dispositivo_padrao(device_id):
    with open("config.json", "w") as config_file:
        json.dump({"dispositivo_padrao": device_id}, config_file)

# Função para carregar o dispositivo padrão
def carregar_dispositivo_padrao():
    try:
        with open("config.json", "r") as config_file:
            config = json.load(config_file)
            return config.get("dispositivo_padrao", None)  # Retorna None se não encontrar
    except FileNotFoundError:
        return None

# resultados recuperados
resposta_recuperada = ""
manifestacao_similar = ""
mensagem_cliente = ""

# Parâmetros de gravação
SAMPLE_RATE = 48000     # Taxa de amostragem (Hz)
BLOCO_DURACAO = 1       # Duração de cada bloco de captura em segundos
gravacao_ativa = False  # Variável de controle para parar a gravação

# Função para salvar o áudio capturado como um arquivo .wav
def salvar_audio(audio, nome_arquivo):
    sf.write(nome_arquivo, audio, SAMPLE_RATE)
    print(f"Áudio salvo como {nome_arquivo}")

# Função para capturar o áudio da saída do sistema em blocos
def capturar_audio_sistema(device_id):
    global gravacao_ativa
    gravacao_ativa = True
    print("Capturando áudio da saída do sistema...")

    audio_total = np.empty((0,), dtype=np.int16)  # Array inicial vazio para 1 canal

    # Usar o dispositivo correto passado como argumento
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, device=device_id, channels=1, dtype='int16', blocksize=2048, latency='low') as stream:
            print("Capturando áudio da Mixagem Estéreo")
            while gravacao_ativa:
                bloco_audio, _ = stream.read(int(BLOCO_DURACAO * SAMPLE_RATE))
                audio_total = np.concatenate((audio_total, bloco_audio.flatten()))
    except Exception as e:
        print(f"Erro durante a captura de áudio: {e}")
        resultado_texto.insert(tk.END, f"Erro na captura de áudio: {e}\n", "resposta_vermelha")

    # salvar_audio(audio_total, 'audio_capturado.wav')  # Salva o áudio
    return audio_total

# Função para transcrever o áudio
def transcrever_audio(audio):
    recognizer = sr.Recognizer()
    audio_data = sr.AudioData(audio.tobytes(), SAMPLE_RATE, 2)
    
    try:
        print("Transcrevendo o áudio...")
        texto = recognizer.recognize_google(audio_data, language='pt-BR')
        print(f"Transcrição: {texto}")
        return texto
    except sr.UnknownValueError:
        return "Não consegui entender o áudio."
    except sr.RequestError as e:
        return f"Erro na requisição: {e}"

import whisper

# # Função para transcrever o áudio usando Whisper
# # Resultado não foi como esperado, desconsiderei o uso
# def transcrever_audio_com_whisper(audio):
#     try:
#         model = whisper.load_model("medium")
#         temp_filename = "temp_audio.wav"

#         # Salva o áudio em um arquivo temporário usando soundfile
#         sf.write(temp_filename, audio, SAMPLE_RATE, subtype='PCM_16')

#         # Transcreve o arquivo de áudio usando Whisper
#         print("Transcrevendo o áudio com Whisper...")
#         result = model.transcribe(temp_filename, language="pt")
#         texto = result['text']
#         print(f"Transcrição: {texto}")

#         # Remove o arquivo temporário após o uso
#         # os.remove(temp_filename)
#         return texto

#     except Exception as e:
#         print(f"Erro ao transcrever com Whisper: {e}")
#         return "Erro ao transcrever o áudio."


# Função para capturar o áudio e transcrever
def capturar_e_transcrever(device_id):
    global gravacao_ativa, mensagem_cliente
    
    audio = capturar_audio_sistema(device_id)
    transcricao = transcrever_audio(audio)
    
    if transcricao and "Erro" not in transcricao:
        resultado_texto.insert(tk.END, f"Transcrição:\n{transcricao}\n")
        mensagem_cliente = transcricao
    else:
        resultado_texto.insert(tk.END, "Erro: Não foi possível transcrever o áudio.\n", "resposta_vermelha")
        mensagem_cliente = ""
    
    return transcricao


# Função para recuperar as informações da base com base na transcrição
def recuperar_informacoes_da_base():
    selecao = lista_dispositivos.curselection()
    if not selecao:
        resultado_texto.insert(tk.END, "Por favor, selecione um dispositivo de áudio.\n", "resposta_vermelha")
        return

    device_info = lista_dispositivos.get(selecao)
    device_id = int(device_info.split(' ')[0])  # Extrai o número do índice

    # Recuperar a transcrição com o dispositivo selecionado
    transcricao = capturar_e_transcrever(device_id)

    transcricao_preprocessada = preprocess_text(transcricao)
    resultado_texto.insert(tk.END, f"\nTranscrição pré-processada:\n{transcricao_preprocessada}\n")

    # Recuperar resposta e manifestação similar do RAG
    global resposta_recuperada, manifestacao_similar
    resposta_recuperada, manifestacao_similar = recuperar_resposta(transcricao)  # Já pré-processado internamente

    # Exibir a manifestação similar e a resposta recuperada
    resultado_texto.insert(tk.END, f"\nMensagem recuperada:\n{manifestacao_similar}\n")
    resultado_texto.insert(tk.END, f"\nResposta recuperada:\n{resposta_recuperada}\n")

# Função para gerar uma nova resposta usando GPT-3.5
def gerar_resposta_com_gpt():
    global resposta_recuperada, manifestacao_similar, mensagem_cliente
    
    if not mensagem_cliente:
        resultado_texto.delete(1.0, tk.END)
        resultado_texto.insert(tk.END, "Por favor, realize a transcrição primeiro.\n", "resposta_vermelha")
        return
    
    # Verificar se a resposta recuperada e a manifestação similar estão definidas
    if not resposta_recuperada or not manifestacao_similar:
        resultado_texto.insert(tk.END, "Erro: Nenhuma manifestação similar foi recuperada.\n", "resposta_vermelha")
        return
    
    # Chamada para gerar a resposta automática com o GPT
    try:
        print(f"Mensagem cliente: {mensagem_cliente}")
        print(f"Manifestação similar: {manifestacao_similar}")
        print(f"Resposta recuperada: {resposta_recuperada}")
        
        # Passar os argumentos necessários para a função gerar_resposta_automatica
        resposta_automatica = gerar_resposta_automatica(mensagem_cliente, resposta_recuperada, manifestacao_similar)
        
        # Exibir a resposta automática
        resultado_texto.insert(tk.END, f"\nResposta sugerida:\n{resposta_automatica}\n", "resposta_verde")

    except Exception as e:
        resultado_texto.insert(tk.END, f"Erro ao gerar a resposta automática: {e}\n", "resposta_vermelha")


def iniciar_recuperacao():
    global mensagem_cliente, resposta_recuperada, manifestacao_similar

    # Verificar se a transcrição foi realizada
    if not mensagem_cliente:
        resultado_texto.delete(1.0, tk.END)
        resultado_texto.insert(tk.END, "Por favor, clique em 'Iniciar transcrição' para realizar a transcrição antes de verificar as similaridades.\n", "resposta_vermelha")
        return
    
    # Recuperar resposta e manifestação similar
    resposta_recuperada, manifestacao_similar = recuperar_resposta(mensagem_cliente)
    
    # Verificação de depuração para verificar o que foi recuperado
    print(f"Manifestação similar recuperada: {manifestacao_similar}")
    print(f"Resposta recuperada: {resposta_recuperada}")
    
    if not resposta_recuperada or not manifestacao_similar:
        resultado_texto.insert(tk.END, "Nenhuma manifestação similar encontrada.\n", "resposta_vermelha")
        return
    
    # Exibir no campo de texto da interface
    resultado_texto.insert(tk.END, f"\nMensagem recuperada:\n{manifestacao_similar}\n")
    resultado_texto.insert(tk.END, f"\nResposta recuperada:\n{resposta_recuperada}\n")


# Função para iniciar o processo de transcrição em uma nova thread
def iniciar_transcricao():
    resultado_texto.delete(1.0, tk.END)
    
    selecao = lista_dispositivos.curselection()
    if not selecao:
        resultado_texto.delete(1.0, tk.END)
        resultado_texto.insert(tk.END, "Por favor, selecione um dispositivo de áudio.\n", "resposta_vermelha")
        return
    
    device_info = lista_dispositivos.get(selecao)
    device_id = int(device_info.split(' ')[0])  # Extrai o número do índice
    
    resultado_texto.insert(tk.END, f"- Dispositivo selecionado: {device_info}\n\n", "info")

    salvar_dispositivo_padrao(device_id)
    
    # Inicia a thread para a captura e transcrição de áudio
    thread = threading.Thread(target=capturar_e_transcrever, args=(device_id,))
    thread.start()

# Função para parar a gravação de áudio
def parar_transcricao():
    global gravacao_ativa
    gravacao_ativa = False
    
    # Exibe a mensagem de interrupção em cinza claro
    resultado_texto.insert(tk.END, "- Gravação interrompida.\n\n- Transcrevendo...\n\n", "info")

# Atualiza a lista de dispositivos ao iniciar a aplicação
def atualizar_dispositivos():
    dispositivos = sd.query_devices()
    lista_dispositivos.delete(0, tk.END)
    dispositivo_padrao = carregar_dispositivo_padrao()

    for i, dispositivo in enumerate(dispositivos):
        lista_dispositivos.insert(tk.END, f"{i} {dispositivo['name']} - {dispositivo['hostapi']} ({dispositivo['max_input_channels']} in, {dispositivo['max_output_channels']} out)")

        # Seleciona o dispositivo padrão, se ele existir
        if dispositivo_padrao is not None and i == dispositivo_padrao:
            lista_dispositivos.select_set(i)

# Função para criar o menu
def criar_menu():
    menubar = tk.Menu(root)
    dispositivos_menu = tk.Menu(menubar, tearoff=0)
    
    # Adiciona a opção de dispositivos de captura no menu
    dispositivos_menu.add_command(label="Atualizar Dispositivos", command=atualizar_dispositivos)

    root.config(menu=menubar)

# ----------------- Interface Gráfica com Tkinter ---------------------
root = tk.Tk()
root.title("Transcrição de Áudio com IA")
root.geometry("600x500")

# Cria o menu
criar_menu()

# Lista de dispositivos de áudio
lista_dispositivos = tk.Listbox(root, height=4, width=90)
lista_dispositivos.pack(padx=10, pady=10)

# Criação de um frame para posicionar os botões lado a lado
frame_botoes = tk.Frame(root)
frame_botoes.pack(pady=10)

# Botão para recuperar informações da base
botao_recuperar = tk.Button(frame_botoes, text="Iniciar transcrição", command=iniciar_transcricao)
botao_recuperar.grid(row=0, column=0, padx=5)

# Botão para parar a transcrição
botao_parar = tk.Button(frame_botoes, text="Parar Transcrição", command=parar_transcricao)
botao_parar.grid(row=0, column=1, padx=5)

# Botão para recuperar informações da base
botao_recuperar = tk.Button(frame_botoes, text="Verificar similaridades", command=iniciar_recuperacao)
botao_recuperar.grid(row=0, column=2, padx=5)

# Botão para gerar resposta com GPT-3.5
botao_gerar_resposta = tk.Button(frame_botoes, text="Gerar Resposta com IA", command=gerar_resposta_com_gpt)
botao_gerar_resposta.grid(row=0, column=3, padx=5)

# Área de texto para exibir as transcrições
resultado_texto = ScrolledText(root, wrap=tk.WORD, height=30, width=70)
resultado_texto.tag_configure("info", foreground="gray")
resultado_texto.tag_configure("resposta_verde", foreground="green")
resultado_texto.tag_configure("resposta_vermelha", foreground="red")
resultado_texto.pack(padx=10, pady=10)


# Atualiza a lista de dispositivos ao iniciar a aplicação
atualizar_dispositivos()

# Inicia o loop do Tkinter
root.mainloop()