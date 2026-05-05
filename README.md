# 🐧 Penguin Classifier - AG2 Inatel

Este projeto foi desenvolvido como parte da disciplina **AG2** (Engenharia de Computação e Software) no **Inatel**. O objetivo é criar um modelo de Machine Learning capaz de classificar três espécies de pinguins do Arquipélago Palmer com base em suas características físicas.

---

## 🚀 O Projeto

O classificador utiliza o dataset `palmerpenguins`, que contém medições de 333 pinguins de três ilhas diferentes da Antártida. O modelo foi treinado para identificar as seguintes espécies:
*   **Adélie** 🐧
*   **Chinstrap** 🐧
*   **Gentoo** 🐧

### 📊 Atributos Utilizados
Para a classificação, o modelo analisa 6 atributos principais:
1. **Island**: Ilha de origem (Biscoe, Dream ou Torgersen).
2. **Sex**: Sexo do pinguim.
3. **Culmen Length**: Comprimento do bico (mm).
4. **Culmen Depth**: Profundidade do bico (mm).
5. **Flipper Length**: Comprimento da nadadeira (mm).
6. **Body Mass**: Massa corporal (g).

---

## 👥 Divisão de Tarefas

O projeto foi dividido de forma modular entre dois integrantes, seguindo as etapas propostas no roteiro oficial:

### 🏗️ Pessoa A: Engenharia de Dados (Passos 1 a 4)
*   Coleta e leitura do conjunto de dados bruto.
*   **Mapeamento Categórico**: Conversão de textos (Ilha, Sexo, Espécie) para números inteiros.
*   **Limpeza e Organização**: Reordenação das colunas e tratamento de valores nulos para garantir a integridade do modelo.

### 🧠 Pessoa B: Ciência de Dados (Passos 5 a 9)
*   **Divisão de Dados**: Separação em 80% para treinamento e 20% para testes.
*   **Modelagem**: Implementação do algoritmo de **Decision Tree** (Árvore de Decisão).
*   **Avaliação**: Geração de métricas de acurácia, `classification_report` e matriz de confusão.
*   **Interface**: Criação da lógica de predição interativa para novos dados inseridos pelo usuário.

---

## 🛠️ Como Executar

1. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

   
