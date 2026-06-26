# 🧪 Checkpoint — Aula 01: Classificador CNN

## Status: ⏸️ Pausado (pausa para almoço)

---

## 📌 O que já foi feito

### Contexto Geral
- Iniciamos o projeto de Classificador de Imagens com CNN + reconhecimento ao vivo via webcam
- Abordagem: **didática socrática** — o Professor guia com perguntas, não dá código pronto
- Ferramentas: TensorFlow, OpenCV, scikit-learn, matplotlib, numpy

### Categorias Escolhidas
1. **Garrafa** (pasta `0/`)
2. **Mouse** (pasta `1/`)
3. **Teclado** (pasta `2/`)

### Material de Referência Fornecido
- `README-aula01.md` na raiz do repo com:
  - Arquitetura visual do sistema
  - Roteiro de pesquisa por etapa (dataset → pré-processamento → CNN → treino → avaliação → webcam)
  - 13 perguntas socráticas para guiar o aprendizado
  - Referências: LeNet-5 (LeCun 1998), Keras docs, PyImageSearch, Goodfellow Cap. 7

### Feedback das Respostas do Aluno
- **Categorias:** boa escolha, discutir ângulo de captura e cor vs cinza
- **Camadas:** intuição "mais ≠ melhor" ✅, mas 8 camadas é demais para 32×32 — são 2-3 na prática
- **"Número par é melhor":** sem base teórica, descartar
- **Overfitting:** o aluno já previu que esse será o principal problema — ponto de partida excelente

---

## 🎯 Próximos Passos (quando retornar)

### Passo 1 — Montar o Dataset
- [ ] Criar estrutura `Imagens/0/`, `Imagens/1/`, `Imagens/2/`
- [ ] Tirar 30-50 fotos de cada objeto (com celular ou webcam)

### Passo 2 — Carregador de Dados
- [ ] Escrever script que carrega as imagens, pré-processa (cinza, 32×32, equalização)
- [ ] Separar treino/validação/teste com train_test_split

### Passo 3 — Arquitetura CNN
- [ ] Definir modelo Sequential (~2-3 camadas convolucionais)
- [ ] Compilar (Adam, sparse_categorical_crossentropy)

### Passo 4 — Treino e Avaliação
- [ ] Treinar com EarlyStopping
- [ ] Gerar gráficos loss × accuracy
- [ ] Diagnóstico de overfitting

### Passo 5 — Webcam ao Vivo
- [ ] Carregar modelo.h5
- [ ] Capturar frame, pré-processar, inferir, exibir

---

## 📦 Estrutura Planejada do Repositório

```
classificador-cnn/
├── .hermes/
│   └── plans/
│       └── CNN-classificador-aula01.md
├── Imagens/
│   ├── 0/   (garrafa)
│   ├── 1/   (mouse)
│   └── 2/   (teclado)
├── README-aula01.md
├── treino.py          # carregamento + CNN + treino
└── webcam.py          # inferência ao vivo
```

---

## 📚 Referências Principais

1. LeCun et al. (1998) — "Gradient-Based Learning Applied to Document Recognition"
   → Arquitetura LeNet-5, Figura 2

2. Documentação Keras:
   - `tf.keras.Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`

3. PyImageSearch (Adrian Rosebrock) — tutoriais de CNN com Keras

4. Goodfellow, Bengio, Courville — "Deep Learning", Cap. 7 (regularização)

---

> *"Você não faz uma pergunta porque espera a resposta — você faz porque merece a resposta."*
