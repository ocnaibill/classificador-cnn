"""
reconhecimento.py — Reconhecimento ao vivo com a webcam.

Carrega o modelo treinado (modelo.h5) e exibe em tempo real
a classe prevista para cada frame da webcam.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from carregar_dados import NOMES_CLASSES, TAMANHO_IMG


# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

CAMINHO_MODELO = "modelo.h5"
CONFIANCA_MINIMA = 0.6  # Só mostra predição se confiança > 60%


# ─────────────────────────────────────────────
# PRÉ-PROCESSAMENTO (igual ao treino!)
# ─────────────────────────────────────────────

def preprocessar_frame(frame):
    """
    Aplica o mesmo pré-processamento usado no treino:
    cinza → redimensionar 32×32 → equalizar → normalizar.
    """
    # Converte para cinza
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimensiona
    img = cv2.resize(img, (TAMANHO_IMG, TAMANHO_IMG))

    # Equaliza histograma
    img = cv2.equalizeHist(img)

    # Normaliza pixels para [0, 1] (float32)
    img = img.astype(np.float32) / 255.0

    # Adiciona dimensões: (32, 32) → (1, 32, 32, 1)
    img = np.expand_dims(img, axis=0)    # batch
    img = np.expand_dims(img, axis=-1)   # canal

    return img


# ─────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("🎥 RECONHECIMENTO AO VIVO — CNN")
    print("=" * 50)

    # 1. Carregar modelo
    print(f"\n📦 Carregando modelo: {CAMINHO_MODELO}...")
    modelo = load_model(CAMINHO_MODELO)
    print("   ✅ Modelo carregado!")

    # 2. Abrir webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Erro ao abrir a webcam.")
        return

    print("\n🎯 Pressione 'q' para sair")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro ao capturar frame.")
            break

        # ── Faz a predição ──
        img_processada = preprocessar_frame(frame)
        predicoes = modelo.predict(img_processada, verbose=0)[0]
        classe_id = np.argmax(predicoes)
        confianca = predicoes[classe_id]
        nome_classe = NOMES_CLASSES[classe_id]

        # ── Prepara texto para exibir ──
        if confianca >= CONFIANCA_MINIMA:
            texto = f"{nome_classe} ({confianca:.0%})"
            cor = (0, 255, 0)  # Verde — confiante
        else:
            texto = "Indefinido"
            cor = (0, 0, 255)  # Vermelho — baixa confiança

        # ── Desenha na tela ──
        # Fundo preto semi-transparente pro texto
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Nome da classe (grande)
        cv2.putText(
            frame, texto, (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, cor, 3,
        )

        # Barra de confiança
        barra_x = 360
        barra_y = 35
        barra_largura = 150
        barra_altura = 20
        largura_preenchida = int(barra_largura * confianca)

        # Fundo da barra
        cv2.rectangle(frame, (barra_x, barra_y),
                      (barra_x + barra_largura, barra_y + barra_altura),
                      (50, 50, 50), -1)
        # Preenchimento
        cv2.rectangle(frame, (barra_x, barra_y),
                      (barra_x + largura_preenchida, barra_y + barra_altura),
                      cor, -1)
        # Bordas
        cv2.rectangle(frame, (barra_x, barra_y),
                      (barra_x + barra_largura, barra_y + barra_altura),
                      (200, 200, 200), 2)

        # ── Exibe ──
        cv2.imshow("Reconhecimento ao Vivo", frame)

        # Sai com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Limpeza ──
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Reconhecimento encerrado.")


if __name__ == "__main__":
    main()
