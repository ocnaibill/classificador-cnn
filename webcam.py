#!/usr/bin/env python3
"""Script para capturar fotos via webcam para dataset de CNN."""

import os
import sys
import cv2
import numpy as np


# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

# Traduz número da pasta → nome visível na tela
# Ordem: 0=copo, 1=mouse, 2=controle
NOMES_CLASSES = ["copo", "mouse", "controle", "fundo"]


# ─────────────────────────────────────────────
# CAPTURA
# ─────────────────────────────────────────────

def capturar(nome_classe, pasta_destino):
    """Abre a webcam e captura frames salvando na pasta_destino."""

    # Garante que a pasta de destino existe
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        sys.exit(1)

    # Conta quantas fotos já existem na pasta (idempotência)
    contador = len(
        [
            f
            for f in os.listdir(pasta_destino)
            if f.endswith(".png") and os.path.isfile(os.path.join(pasta_destino, f))
        ]
    )

    print(f"Iniciando captura para a classe '{nome_classe}'.")
    print(f"Fotos já existentes na pasta: {contador}")
    print("Comandos: [ENTER] tira foto  |  [P] sai do programa")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao ler frame da webcam.")
            break

        # Cópia para desenhar texto sem sujar o frame original que será salvo
        frame_com_texto = frame.copy()

        # ── Informações na tela ──
        texto = f"Classe: {nome_classe}  |  Fotos: {contador}"
        cv2.putText(
            frame_com_texto, texto, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        cv2.putText(
            frame_com_texto, "[ENTER] Capturar  —  [P] Sair", (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        cv2.imshow("Captura de Fotos", frame_com_texto)

        # ── Aguarda tecla ──
        tecla = cv2.waitKey(1) & 0xFF

        # ENTER → salva frame
        if tecla == 13:
            nome_arquivo = f"foto_{contador}.png"
            caminho_completo = os.path.join(pasta_destino, nome_arquivo)

            cv2.imwrite(caminho_completo, frame)
            print(f"✅ Foto {contador} salva: {caminho_completo}")

            contador += 1

            # ── Flash verde de feedback ──
            flash = np.full_like(frame, (0, 255, 0))
            cv2.imshow("Captura de Fotos", flash)
            cv2.waitKey(80)

        # P → sai
        elif tecla == ord("p") or tecla == ord("P"):
            print("Encerrando captura...")
            break

    # ── Limpeza ──
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Captura finalizada. Total de fotos em '{pasta_destino}': {contador}")


# ─────────────────────────────────────────────
# EXECUÇÃO
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Erro: informe a classe a ser capturada (0, 1 ou 2)")
        print("Uso: python webcam.py <classe>")
        print(f"    0 = {NOMES_CLASSES[0]}")
        print(f"    1 = {NOMES_CLASSES[1]}")
        print(f"    2 = {NOMES_CLASSES[2]}")
        sys.exit(1)

    try:
        classe = int(sys.argv[1])
    except ValueError:
        print("Erro: o argumento deve ser um número inteiro (0, 1 ou 2)")
        sys.exit(1)

    if classe < 0 or classe > 2:
        print(f"Erro: classe inválida. Opções: 0={NOMES_CLASSES[0]}, "
              f"1={NOMES_CLASSES[1]}, 2={NOMES_CLASSES[2]}")
        sys.exit(1)

    pasta_destino = f"Imagens/{classe}"
    nome_classe = NOMES_CLASSES[classe]

    print(f"📸 Classe selecionada: {classe} — {nome_classe}")
    print(f"📁 Pasta destino: {pasta_destino}")
    print()

    capturar(nome_classe, pasta_destino)


if __name__ == "__main__":
    main()
