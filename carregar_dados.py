"""
carregar_dados.py — Carregamento e pré-processamento do dataset.

Percorre as pastas Imagens/0, Imagens/1, Imagens/2, lê cada foto,
pré-processa (cinza, 32×32, equalização, normalização) e separa
em treino, validação e teste.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

RAIZ_IMAGENS = "Imagens"
NOMES_CLASSES = ["copo", "mouse", "controle"]
TAMANHO_IMG = 32
TESTE_TAMANHO = 0.2       # 20% dos dados vão pra teste
VALIDACAO_TAMANHO = 0.15  # 15% do restante vão pra validação
SEMENTE_ALEATORIA = 42    # Para reprodutibilidade


# ─────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────

def carregar_dados():
    """
    Carrega todas as imagens das pastas numeradas, pré-processa e
    retorna os conjuntos de treino, validação e teste.

    Retorna
    -------
    X_train, X_val, X_test : np.ndarray
        Imagens pré-processadas (forma: (N, 32, 32, 1)).
    y_train, y_val, y_test : np.ndarray
        Rótulos inteiros (0, 1, 2) correspondentes a cada imagem.
    """
    X = []   # Lista de imagens (dados de entrada)
    y = []   # Lista de rótulos (classes)

    # Percorre as pastas Imagens/0, Imagens/1, Imagens/2
    for classe_id in range(len(NOMES_CLASSES)):
        caminho_pasta = os.path.join(RAIZ_IMAGENS, str(classe_id))

        if not os.path.exists(caminho_pasta):
            print(f"⚠️  Pasta {caminho_pasta} não encontrada. Pulando.")
            continue

        # Lista apenas arquivos .png
        arquivos = sorted([
            f for f in os.listdir(caminho_pasta)
            if f.endswith(".png") and os.path.isfile(os.path.join(caminho_pasta, f))
        ])

        print(f"📂 {NOMES_CLASSES[classe_id]} — {len(arquivos)} imagens")

        for nome_arquivo in arquivos:
            caminho_completo = os.path.join(caminho_pasta, nome_arquivo)

            # 1. Lê a imagem
            img = cv2.imread(caminho_completo)

            if img is None:
                print(f"    ⚠️  Erro ao ler {caminho_completo} — pulando")
                continue

            # 2. Converte para escala de cinza
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 3. Redimensiona para 32×32
            img = cv2.resize(img, (TAMANHO_IMG, TAMANHO_IMG))

            # 4. Equaliza o histograma (corrige iluminação)
            img = cv2.equalizeHist(img)

            # 5. Normaliza pixels para o intervalo [0, 1]
            img = img.astype(np.float32) / 255.0

            # 6. Adiciona dimensão do canal (32, 32) → (32, 32, 1)
            img = np.expand_dims(img, axis=-1)

            X.append(img)
            y.append(classe_id)

    # Converte listas para arrays do NumPy
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"\n📊 Total de imagens carregadas: {len(X)}")
    print(f"   Formato dos dados (X): {X.shape}")
    print(f"   Formato dos rótulos (y): {y.shape}")

    # ── Separação treino / validação / teste ──

    # Primeiro: separa 20% para teste, 80% para treino+validação
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TESTE_TAMANHO,
        random_state=SEMENTE_ALEATORIA,
        stratify=y,  # Mantém proporção das classes
    )

    # Depois: dos 80% restantes, separa 15% para validação
    # (15% de 80% = 12% do total; treino = 68% do total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VALIDACAO_TAMANHO,
        random_state=SEMENTE_ALEATORIA,
        stratify=y_temp,
    )

    print(f"\n📦 Divisão dos dados:")
    print(f"   Treino:     {len(X_train)} imagens ({len(X_train)/len(X)*100:.0f}%)")
    print(f"   Validação:  {len(X_val)} imagens ({len(X_val)/len(X)*100:.0f}%)")
    print(f"   Teste:      {len(X_test)} imagens ({len(X_test)/len(X)*100:.0f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────
# TESTE RÁPIDO (execução direta)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = carregar_dados()
