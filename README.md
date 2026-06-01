# 🐧 Penguin Classifier — AG2 Inatel

Este projeto foi desenvolvido como parte da disciplina **AG2** (Engenharia de Computação e Software) no **Inatel**. O objetivo é criar um modelo de Machine Learning capaz de classificar três espécies de pinguins do Arquipélago Palmer com base em suas características físicas.

---

## 🚀 O Projeto

O classificador utiliza o dataset `palmerpenguins`, que contém medições de 333 pinguins de três ilhas diferentes da Antártida. O modelo foi treinado para identificar as seguintes espécies:

- **Adélie** 🐧
- **Chinstrap** 🐧
- **Gentoo** 🐧

### 📊 Atributos Utilizados

Para a classificação, o modelo analisa 6 atributos principais:

| # | Atributo | Descrição |
|---|----------|-----------|
| 1 | **Island** | Ilha de origem (Torgersen, Biscoe ou Dream) |
| 2 | **Culmen Length** | Comprimento do bico (mm) |
| 3 | **Culmen Depth** | Profundidade do bico (mm) |
| 4 | **Flipper Length** | Comprimento da nadadeira (mm) |
| 5 | **Body Mass** | Massa corporal (g) |
| 6 | **Sex** | Sexo do pinguim |

---

## 👥 Divisão de Tarefas

### 🏗️ Pessoa A(Mateus Augusto De Faria(290) — Engenharia de Dados (Passos 1 a 4)

- Coleta e leitura do dataset bruto (`palmerpenguins.csv`)
- Mapeamento categórico: conversão de texto para inteiros
- Limpeza: remoção de valores nulos
- Reordenação das colunas e exportação do `palmerpenguins_final.csv`

### 🧠 Pessoa B(Antonio Augusto D'Assumpção(221) — Ciência de Dados (Passos 5 a 9)

- **Passo 5:** Divisão dos dados em 80% treino e 20% teste
- **Passo 6:** Escolha e instanciação do modelo Decision Tree
- **Passo 7:** Treinamento com `fit()` e predição com `predict()`
- **Passo 8:** Métricas com `classification_report` e Matriz de Confusão
- **Passo 9:** Interface interativa para classificar novos pinguins

---

## 📁 Estrutura do Projeto

```
penguin-classifier-ag2/
├── model/
│   ├── classifier.py               ← Pessoa B (Passos 5–9)
│   ├── main.ipynb                  ← Pessoa A (Passos 1–4)
│   ├── palmerpenguins.csv          ← Dataset original
│   └── palmerpenguins_final.csv    ← Dataset processado pela Pessoa A
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Como Executar

**1. Instale as dependências:**
```bash
pip install -r requirements.txt
```

**2. Rode o classificador:**
```bash
python model/classifier.py
```

**3. Ao final, a interface pedirá dados de um pinguim para classificar:**
```
Ilha (0=Torgersen | 1=Biscoe | 2=Dream): 0
Sexo (0=MALE | 1=FEMALE): 0
Comprimento do cúlmen (mm): 39.1
Profundidade do cúlmen (mm): 18.7
Comprimento da nadadeira (mm): 181
Massa corporal (g): 3750

  Espécie prevista: Adelie
```

---

## 📈 Resultados

Acurácia obtida com 67 amostras de teste:

| Espécie | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Adelie | 0.93 | 0.90 | 0.91 |
| Chinstrap | 0.82 | 1.00 | 0.90 |
| Gentoo | 1.00 | 0.92 | 0.96 |
| **Geral** | **0.93** | **0.93** | **0.93** |

---

## 🗂️ Mapeamento Numérico

| Coluna | Valor Original | Valor Numérico |
|--------|---------------|----------------|
| island | Torgersen | 0 |
| island | Biscoe | 1 |
| island | Dream | 2 |
| sex | MALE | 0 |
| sex | FEMALE | 1 |
| species | Adelie | 0 |
| species | Chinstrap | 1 |
| species | Gentoo | 2 |
