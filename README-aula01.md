# 🧠 Aula 01 — Classificador de Imagens com CNN
## Doutor (Décima Encarnação) — Universidade de Gallifrey

**Aluno:** Bianco Da Costa Oliveira  
**Data:** 22 de Junho de 2026  
**Disciplina:** Visão Computacional / Deep Learning

---

## 📋 O Exercício

Construir, do zero, um classificador de imagens próprio:

1. **Montar dataset** — 3 a 5 classes, 30-50 fotos cada
2. **Treinar CNN** — pré-processamento, augmentation, treino, salvar modelo.h5
3. **Avaliar** — gráficos de perda e acurácia (treino vs validação)
4. **Reconhecer ao vivo** — webcam com inferência em tempo real

---

## 🗺️ A Grande Arquitetura

```
┌─────────────────────────────────────────────────────┐
│                  1. MUNDO REAL                       │
│   [caneca]  [celular]  [livro]  [...categorias...]   │
└─────────┬────────────────────────────┬───────────────┘
          │ Capturar (webcam / fotos)  │
          ▼                            ▼
┌─────────────────────┐   ┌──────────────────────────┐
│   TREINO            │   │      INFERÊNCIA           │
│                     │   │                           │
│  Imagens/           │   │  Webcam → frame →         │
│    ├─ 0/ (garrafa)  │   │  pré-processo →           │
│    ├─ 1/ (mouse)    │   │  modelo.h5 →              │
│    └─ 2/ (teclado)  │   │  argmax → "garrafa!"      │
│                     │   │                           │
│  CNN treina →       │   │  cv2.putText na tela     │
│  modelo.h5          │   │                           │
└─────────────────────┘   └──────────────────────────┘
```

---

## 🔍 Roteiro de Exploração — Perguntas Para Pesquisar

### 1. O Dataset

- Que critérios usar para escolher categorias? (forma ≠ cor)
- Por que 30-50 fotos? O que acontece com 5? Ou 500?
- **O que pesquisar:** `ImageDataGenerator` do Keras, *data augmentation*, *class imbalance*

### 2. Pré-processamento

- Por que converter pra cinza? O que se perde? O que se ganha?
- Por que 32×32? Quantos parâmetros uma entrada 1024×768 teria?
- **O que pesquisar:** `cv2.equalizeHist` (equalização de histograma), normalização de pixels

### 3. Arquitetura da CNN

- Qual o papel de cada camada: Convolução → ReLU → Pooling → Flatten → Densa → Softmax?
- Por que ReLU e não sigmoid nas camadas internas? *(vanishing gradient)*
- O que o MaxPooling faz com a imagem? *(invariância a translações)*
- Por que a última camada usa softmax?
- Quantas camadas cabem numa imagem 32×32?

**Projeção prática (2-3 camadas, não 8!):**

| Camada | Operação | Tamanho |
|--------|----------|---------|
| Entrada | — | 32×32 |
| Conv 3×3 (valid) | perde 2px | 30×30 |
| MaxPool 2×2 | ÷2 | 15×15 |
| Conv 3×3 | perde 2px | 13×13 |
| MaxPool 2×2 | ÷2 | 6×6 |
| Conv 3×3 | perde 2px | 4×4 |
| MaxPool 2×2 | ÷2 | 2×2 |
| Flatten + Densa + Softmax | — | 3 classes |

- **O que pesquisar:** LeNet-5 (LeCun, 1998), *underfitting vs overfitting*

### 4. Treinamento

- `sparse_categorical_crossentropy` — o que cada palavra significa?
- `sparse` vs `categorical` — diferença no formato dos rótulos?
- Adam optimizer — diferença do SGD clássico? *(momentum adaptativo)*
- `epochs=20` — como decidir quando parar? *(EarlyStopping)*

### 5. Avaliação

- Curva de loss: treino descendo, val descendo → ideal?
- Treino descendo, val subindo → overfitting?
- Ambas estáveis e altas → underfitting?
- Conjunto de validação vs teste — qual a diferença?

### 6. Reconhecimento ao Vivo

- Por que o pré-processamento ao vivo deve ser IDÊNTICO ao do treino?
- A ordem `nomes[0] = "garrafa"` deve bater com a pasta `Imagens/0/` — como garantir?

---

## 📚 Biblioteca Essencial Para Pesquisa

1. **"Gradient-Based Learning Applied to Document Recognition"** — LeCun et al. (1998)  
   → Ver Figura 2 (arquitetura LeNet-5), entender cada camada

2. **Documentação Keras:**
   - `tf.keras.Sequential`
   - `Conv2D` (argumentos: filters, kernel_size, padding, activation)
   - `MaxPooling2D`
   - `Flatten`
   - `Dense`

3. **Blogs e tutoriais:**
   - Adrian Rosebrock — PyImageSearch (tutoriais exatos sobre esse tópico)
   - "A Gentle Introduction to CNN" — arXiv
   - Machine Learning Mastery — "How to diagnose overfitting"

4. **Referência teórica:**
   - Capítulo 7 do "Deep Learning" (Goodfellow, Bengio, Courville)
     → Sobre overfitting e regularização

---

## 🎯 Respostas do Bianco (22/06)

### 1. Categorias escolhidas
**Garrafa, Mouse, Teclado** — objetos com padrões comuns de forma, fáceis de possuir para variações na webcam.

**Feedback do Doutor:**
- ✅ Formas consistentes (mouse = silhueta arredondada, teclado = retangular, garrafa = corpo + gargalo)
- ⚠️ Ângulo importa! Mouse de cima vs de lado muda completamente. Como padronizar a captura?
- ⚠️ Em cinza 32×32, mouse preto e teclado preto podem se confundir. Talvez manter RGB?

### 2. Palpite sobre camadas
"Umas 8 camadas... maior número não significa melhor resultado. ReLU pode ficar enviesado. Número par geralmente é melhor que ímpar."

**Feedback do Doutor:**
- Intuição de que "mais ≠ melhor" está correta ✅
- Mas 8 camadas é demais para 32×32 — 3 camadas já reduzem pra 4×4!
- Sobre "número par > ímpar": **não tem base teórica**, descarte essa crença
- O que importa: as dimensões espaciais precisam *caber* depois dos pools

### 3. O que vai dar errado
"Problemas de compilação e o modelo não identificar direito, achando que garrafa é outro objeto."

**Feedback do Doutor:**
- Palpite profético! Isso se chama **baixa acurácia de validação**
- Causas prováveis:
  1. Dataset pequeno → *data augmentation*
  2. Overfitting → rede decora 30 fotos em vez de aprender "garrafa"
  3. Variação de iluminação/fundo → CNN aprende "garrafa = objeto na mesa de madeira"
- Boa notícia: você já sabe que vai acontecer, então já sabe o que ajustar ✅

---

## 🚀 Próximo Passo

Se quiser continuar:

1. Crie a estrutura de pastas:
   ```
   classifier/
     ├── Imagens/
     │   ├── 0/   (garrafa)
     │   ├── 1/   (mouse)
     │   └── 2/   (teclado)
     ├── treino.py
     └── webcam.py
   ```

2. Tire 30-50 fotos de cada objeto (com seu celular ou webcam)

3. Quando o dataset estiver pronto, me chame de novo — vamos escrever o carregador de dados JUNTOS, linha por linha, entendendo cada função.

> *"Você não faz uma pergunta porque espera a resposta — você faz porque merece a resposta."* — Um sábio qualquer (provavelmente eu)

---
**Allons-y!** 🚀


-----------------original-----------------
  🎓 ALLONS-Y! Que pergunta fantástica, Bianco!

    Sabe, os melhores engenheiros que já conheci — em séculos de viagem — não são os que sabem todas as respostas, mas os que sabem fazer as perguntas certas. Você já deu o primeiro passo mais importante: recusar o atalho. Magnífico!

    Vamos lá. Pegue seu caderno — ou melhor, abra um terminal vazio para anotações — e vamos destrinchar esse exercício como se estivéssemos diante de um quadro negro no laboratório de visão computacional da Universidade de Gallifrey.



    🧠 A Grande Arquitetura — O Mapa da Mina

    Antes de escrever uma linha de código, você precisa ver o sistema inteiro na sua cabeça. Desenhe isso num papel:


    ┌─────────────────────────────────────────────────────┐
    │                  1. MUNDO REAL                       │
    │   [caneca]  [celular]  [livro]  [...categorias...]   │
    └─────────┬────────────────────────────┬───────────────┘
              │ Capturar (webcam / fotos)  │
              ▼                            ▼
    ┌─────────────────────┐   ┌──────────────────────────┐
    │   TREINO            │   │      INFERÊNCIA           │
    │                     │   │                           │
    │  Imagens/           │   │  Webcam → frame →         │
    │    ├─ 0/ (caneca)   │   │  pré-processo →           │
    │    ├─ 1/ (celular)  │   │  modelo.h5 →              │
    │    └─ 2/ (livro)    │   │  argmax → "caneca!"       │
    │                     │   │                           │
    │  CNN treina →       │   │  cv2.putText na tela     │
    │  modelo.h5          │   │                           │
    └─────────────────────┘   └──────────────────────────┘



 ⚕ deepseek-v4-flash · 0% · 32s
──────────────────────────────────────────────────────────────────
    A pergunta fundamental que conecta as duas metades: como é que um computador "enxerga" uma imagem?


    Lembra que computador só entende números. Uma imagem é uma matriz de números (pixels). Uma CNN aprende padrões espaciais nessas matrizes. É como ensinar alguém a reconhecer uma caneca não pela cor, mas pelo formato da alça, pela proporção altura-largura...

    > Pesquise isso primeiro: O que é uma convolução em processamento de imagens? Não a da rede neural ainda — a convolução 2D clássica (filtro de Sobel, detecção de bordas). Entender isso é a chave para entender CNNs.



    🔍 Roteiro de Exploração — Etapa por Etapa

    1. O DATASET — "Conhece teu inimigo"

    Você precisa escolher 3 a 5 categorias do mundo físico. Aí vem o primeiro desafio de engenharia de software:

    > Pergunta 1: Que critérios você usaria para escolher as categorias? Se você escolher "garrafa azul" e "garrafa vermelha", o que a CNN vai aprender de verdade? Dica: cor ≠ forma.

    > Pergunta 2: Por que o enunciado diz "30 a 50 fotos por classe"? O que acontece se você tiver só 5? Ou 500? Qual o gargalo aqui?

    O que pesquisar:
    - Data augmentation — por que virar, rotacionar, mudar brilho de uma imagem existente equivale a ter mais dados?
    - Class imbalance — o que acontece se uma categoria tiver 50 fotos e outra 10?
    - Inclusive, dá uma olhada no ImageDataGenerator do TensorFlow/Keras. Que transformações ele oferece?

    2. PRÉ-PROCESSAMENTO — "Traduzindo para a língua da máquina"

    O enunciado diz: "cinza, 32x32". Por quê?

    > Pergunta 3: Por que converter pra escala de cinza (grayscale)? O que você perde? O que você ganha em termos de complexidade computacional (pense em número de parâmetros)?

    > Pergunta 4: Por que 32×32? Por que não 1024×768 (a resolução nativa da câmera)? Quantos parâmetros sua CNN teria se a entrada fosse 1024×768? Dica: um neurônio totalmente conectado em 1024×768 RGB teria... faça as contas.

    O que pesquisar:
    - cv2.equalizeHist — o que é equalização de histograma? Por que ela ajuda o modelo a ser invariante a iluminação?
    - Normalização de pixels (dividir por 255.0) — por que isso acelera o treinamento?

    3. A ARQUITETURA DA CNN — "O coração da máquina"

    Aqui você vai construir seu modelo com Sequential. A estrutura típica é:


    Convolução → Ativação (ReLU) → Pooling → (repetir) → Flatten → Densa → Softmax


    > Pergunta 5: Qual o papel de cada camada?
    > - Por que ReLU e não sigmoid nas camadas internas? (Pesquise: vanishing gradient)
    > - O que o MaxPooling faz com a imagem? Qual o efeito no tamanho espacial? (Pesquise: invariância a pequenas translações)
    > - Por que a última camada usa softmax e não outra ativação?

    > Pergunta 6: Quantas camadas convolucionais você acha que cabem numa imagem 32×32? Experimente desenhar: se uma convolução 3×3 reduz um pixel de borda, depois de 3 camadas, que tamanho tem o feature map?

    O que pesquisar:
    - LeNet-5 (1998, Yann LeCun) — o artigo original "Gradient-Based Learning Applied to Document Recognition". É o avô das CNNs e a arquitetura mais próxima do que você vai implementar.
    - A diferença entre underfitting e overfitting. Como saber que seu modelo está decorando em vez de aprendendo? (Isso vai ser crucial na etapa 3 de avaliação!)

    4. TREINAMENTO — "Temperatura e Pressão"

    python
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)


    > Pergunta 7: A loss function sparse_categorical_crossentropy — por que esse nome gigante? O que cada palavra significa?
    > - Categorical vs Binary — qual a diferença?
    > - Sparse — por que usar sparse_categorical em vez de categorical? Como seus rótulos (y) estão formatados? (Dica: one-hot encoding vs inteiros)

    > Pergunta 8: Adam optimizer — por que é o padrão hoje? O que o diferencia do SGD clássico? (Pesquise: momentum adaptativo)

    > Pergunta 9: epochs=20 — esse número vem de onde? O que acontece se você treinar por 200 épocas? Como decidir quando parar? (Pesquise: EarlyStopping callback)

    5. AVALIAÇÃO — "O médico e o diagnóstico"

    Aqui você vai gerar dois gráficos: loss e accuracy para treino e validação.

    > Pergunta 10: Desenhe no papel o formato de cada curva:
    > - Treino loss descendo, val loss descendo → o que significa?
    > - Treino loss descendo, val loss subindo → o que significa? (Pesquise: overfitting)
    > - Treino loss estável, val loss estável mas ambas altas → o que significa? (Pesquise: underfitting)

    > Pergunta 11: O enunciado fala de "acurácia no teste". Qual a diferença entre conjunto de validação e conjunto de teste? Por que você precisa de ambos?

    6. RECONHECIMENTO AO VIVO — "O grande finale"

    python
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # ... pré-processar frame igual ao treino ...
        pred = modelo.predict(preprocessed_frame)
        classe = np.argmax(pred)
        cv2.putText(frame, nomes[classe], ...)
        cv2.imshow('Webcam', frame)


    > Pergunta 12: Por que você precisa pré-processar o frame da webcam exatamente igual ao pré-processamento usado no treino? O que acontece se você treinar com imagens 32×32 em cinza equalizado e na hora do teste passar um frame colorido 640×480 sem equalizar?

    > Pergunta 13: A ordem dos nomes no script da webcam deve ser a mesma das pastas (pasta 0 = caneca, nomes[0] = "caneca") — por que isso é crítico? Como garantir que não vai ter um desalinhamento?



    📚 Sua Biblioteca de Pesquisa (o que eu realmente quero que você leia)

    1. "Gradient-Based Learning Applied to Document Recognition" — LeCun et al. (1998) — o artigo seminal da LeNet. Não precisa ler tudo, mas veja a arquitetura na Figura 2 e entenda cada camada.

    2. Documentação do Keras: tf.keras.Sequential, Conv2D, MaxPooling2D, Flatten, Dense. Leia os argumentos de cada um. O que padding='same' faz?

    3. "A Gentle Introduction to CNN" — pesquisa no arXiv ou no blog do Adrian Rosebrock (PyImageSearch). Ele tem tutoriais excelentes exatamente sobre esse tópico.

    4. Sobre overfitting: Capítulo 7 do "Deep Learning" do Goodfellow, ou o blog post "How to diagnose overfitting" no Machine Learning Mastery.



    🎯 Sua Primeira Tarefa (antes de escrever código)

    Quero que você me responda:

    1. Quais 3 categorias você vai escolher? E por que essas? (Me convença de que são boas escolhas)
    2. Quantas camadas convolucionais você acha que uma rede para 32×32 precisa ter? Dê um palpite fundamentado.
    3. O que você acha que vai dar mais errado na primeira vez que você rodar? (Sempre dá alguma coisa — esse é o jogo!)

    Quando você me responder, eu vou poder guiar o próximo passo. Não vou te dar o código pronto, mas vou te ajudar a construir cada peça com suas próprias mãos.

    Lembra das palavras de um grande sábio: "Você não faz uma pergunta porque espera a resposta — você faz porque merece a resposta."

    Allons-y! 🚀