"""
treino.py — Treinamento da CNN para classificação de imagens.

Carrega os dados, constrói a rede convolucional, treina, gera
gráficos de acompanhamento e salva o modelo treinado.
"""

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping

from carregar_dados import carregar_dados, NOMES_CLASSES


# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────

EPOCAS = 50
LOTE = 16         # Batch size pequeno (poucos dados)
TAXA_APRENDIZADO = 0.001
PACIENCIA_EARLY = 10  # Se não melhorar em 10 épocas, para


# ─────────────────────────────────────────────
# ARQUITETURA DA CNN
# ─────────────────────────────────────────────

def construir_modelo():
    """
    CNN simples para 32×32×1 com 3 classes.

    Arquitetura:
      Conv(16) → Pool → Conv(32) → Pool → Conv(64) → Pool
      → Flatten → Dense(128) → Dropout(50%) → Dense(3, softmax)
    """
    modelo = Sequential([

        # ── Bloco 1: bordas e texturas simples ──
        Conv2D(16, (3, 3), activation="relu", padding="same",
               input_shape=(32, 32, 1)),
        MaxPooling2D((2, 2)),   # 32 → 16

        # ── Bloco 2: padrões médios ──
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),   # 16 → 8

        # ── Bloco 3: padrões abstratos ──
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),   # 8 → 4

        # ── Classificador ──
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),           # Desliga metade dos neurônios (anti-overfitting)
        Dense(len(NOMES_CLASSES), activation="softmax"),
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return modelo


# ─────────────────────────────────────────────
# GRÁFICOS
# ─────────────────────────────────────────────

def plotar_historico(historico):
    """Gera gráficos de loss e acurácia ao longo do treino."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(historico.history["loss"], label="Treino", linewidth=2)
    ax1.plot(historico.history["val_loss"], label="Validação", linewidth=2)
    ax1.set_title("Perda (Loss)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Acurácia
    ax2.plot(historico.history["accuracy"], label="Treino", linewidth=2)
    ax2.plot(historico.history["val_accuracy"], label="Validação", linewidth=2)
    ax2.set_title("Acurácia")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("grafico_treino.png", dpi=150)
    plt.show()
    print("📊 Gráfico salvo como 'grafico_treino.png'")


# ─────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("📸 CLASSIFICADOR DE IMAGENS — CNN")
    print("=" * 50)

    # 1. Carregar dados
    print("\n📂 Carregando dados...")
    X_train, X_val, X_test, y_train, y_val, y_test = carregar_dados()

    # 2. Construir modelo
    print("\n🏗️  Construindo modelo...")
    modelo = construir_modelo()
    modelo.summary()

    # 3. Callback: para de treinar quando a validação parar de melhorar
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PACIENCIA_EARLY,
        restore_best_weights=True,
        verbose=1,
    )

    # 4. Treinar
    print("\n🎯 Iniciando treinamento...\n")
    historico = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCAS,
        batch_size=LOTE,
        callbacks=[early_stop],
        verbose=1,
    )

    # 5. Gráficos
    plotar_historico(historico)

    # 6. Avaliar no teste
    print("\n🧪 Avaliando no conjunto de teste...")
    perda_teste, acuracia_teste = modelo.evaluate(X_test, y_test, verbose=0)
    print(f"   Loss no teste:  {perda_teste:.4f}")
    print(f"   Acurácia no teste: {acuracia_teste:.2%}")

    # 7. Salvar modelo
    modelo.save("modelo.h5")
    print("\n💾 Modelo salvo como 'modelo.h5'")

    print("\n✅ Treino concluído!")


if __name__ == "__main__":
    main()
