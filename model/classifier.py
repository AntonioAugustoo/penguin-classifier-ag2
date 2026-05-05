"""
AG2 - Inatel | Classificador de Pinguins

Para integrar com a Pessoa A, substitua a seção "MOCK DATA"
por pd.read_csv('penguins_processed.csv') conforme indicado.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# ─────────────────────────────────────────────
# MAPEAMENTO INVERSO  (número → nome da espécie)
# ─────────────────────────────────────────────
SPECIES_MAP = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
ISLAND_MAP  = {"Biscoe": 0, "Dream": 1, "Torgersen": 2}
SEX_MAP     = {"FEMALE": 0, "MALE": 1}

# Ordem das colunas 
FEATURE_COLUMNS = [
    "island",
    "sex",
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
TARGET_COLUMN = "species"


# ════════════════════════════════════════════════════════════
# [PONTO DE INTEGRAÇÃO]
# Quando a Pessoa A terminar, substitua a função abaixo por:
#
#   def carregar_dados():
#       return pd.read_csv("penguins_processed.csv")
#

# ════════════════════════════════════════════════════════════
def carregar_dados() -> pd.DataFrame:
    """
    Retorna um DataFrame simulando o resultado do Passo 4 da Pessoa A.

    Colunas (ordem exigida pelo PDF):
        island | sex | culmen_length_mm | culmen_depth_mm |
        flipper_length_mm | body_mass_g | species

    Mapeamento numérico (conforme PDF):
        island:  Biscoe=0, Dream=1, Torgersen=2
        sex:     FEMALE=0, MALE=1
        species: Adelie=0, Chinstrap=1, Gentoo=2
    """
    dados_mock = {
        # island: 0=Biscoe, 1=Dream, 2=Torgersen
        "island":            [0, 1, 2, 0, 1, 2, 0, 1, 0, 2],
        # sex: 0=FEMALE, 1=MALE
        "sex":               [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        # Atributos físicos
        "culmen_length_mm":  [39.1, 46.5, 38.9, 45.2, 49.3, 37.8, 47.6, 50.1, 42.0, 35.5],
        "culmen_depth_mm":   [18.7, 17.9, 17.8, 14.2, 18.7, 19.0, 14.5, 19.5, 20.2, 16.3],
        "flipper_length_mm": [181,  182,  181,  215,  195,  174,  215,  182,  190,  178 ],
        "body_mass_g":       [3750, 3800, 3625, 5200, 3650, 3200, 5050, 3900, 4300, 3050],
        # species: 0=Adelie, 1=Chinstrap, 2=Gentoo
        "species":           [0,    1,    0,    2,    1,    0,    2,    1,    0,    0   ],
    }
    return pd.DataFrame(dados_mock)


# ─────────────────────────────────────────────
# PASSO 5 — Divisão treino / teste (80% / 20%)
# ─────────────────────────────────────────────
def dividir_dados(df: pd.DataFrame):
    """
    Separa features (X) e alvo (y), depois divide em treino e teste.
    Retorna: X_train, X_test, y_train, y_test
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

  
    n_classes   = y.nunique()
    n_test_min  = max(1, int(len(y) * 0.2))
    usar_strat  = n_test_min >= n_classes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if usar_strat else None
    )

    print(f"[Passo 5] Dataset dividido:")
    print(f"  Treino : {len(X_train)} amostras ({len(X_train)/len(df):.0%})")
    print(f"  Teste  : {len(X_test)} amostras ({len(X_test)/len(df):.0%})\n")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# PASSOS 6 e 7 — Modelo, treinamento e predição
# ─────────────────────────────────────────────
def treinar_modelo(X_train, y_train) -> DecisionTreeClassifier:
    """
    Passo 6: Instancia o modelo Decision Tree.
    Passo 7: Treina com fit() nos dados de treino.
    """
    modelo = DecisionTreeClassifier(
        criterion="gini",    # critério de divisão
        max_depth=5,         # evita overfitting
        random_state=42
    )
    modelo.fit(X_train, y_train)
    print("[Passos 6-7] Modelo Decision Tree treinado com sucesso.\n")
    return modelo


def predizer(modelo: DecisionTreeClassifier, X_test):
    """Passo 7: Classifica as amostras do conjunto de teste."""
    return modelo.predict(X_test)


# ─────────────────────────────────────────────
# PASSO 8 — Métricas de avaliação
# ─────────────────────────────────────────────
def exibir_metricas(y_test, y_pred, modelo, X_train, y_train) -> None:
    """
    Exibe classification_report e Matriz de Confusão.
    Ambos serão úteis para explicar no vídeo de 7 minutos.
    """
    nomes_especies = list(SPECIES_MAP.values())  # ["Adelie", "Chinstrap", "Gentoo"]


    print("[Passo 8] Classification Report:")
    print("─" * 55)
    print(
        classification_report(
            y_test, y_pred,
            labels=[0, 1, 2],
            target_names=nomes_especies,
            zero_division=0
        )
    )

    # — Matriz de Confusão —
    
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AG2 Inatel — Avaliação do Classificador de Pinguins", fontsize=13, fontweight="bold")

    # Matriz de Confusão (contagens absolutas)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nomes_especies).plot(
        cmap="Blues", ax=axes[0]
    )
    axes[0].set_title("Matriz de Confusão (contagens)")

    # Matriz de Confusão (normalizada — mostra proporções)
    cm_norm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2], normalize="true")
    ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=nomes_especies).plot(
        cmap="Greens", ax=axes[1]
    )
    axes[1].set_title("Matriz de Confusão (normalizada)")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Passo 8] Matriz de Confusão salva como 'confusion_matrix.png'\n")

    # — Visualização da Árvore de Decisão —
    fig2, ax2 = plt.subplots(figsize=(18, 8))
    plot_tree(
        modelo,
        feature_names=FEATURE_COLUMNS,
        class_names=nomes_especies,
        filled=True,
        rounded=True,
        ax=ax2
    )
    ax2.set_title("Estrutura da Árvore de Decisão", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("decision_tree.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Passo 8] Árvore de Decisão salva como 'decision_tree.png'\n")


# ─────────────────────────────────────────────
# PASSO 9 — Interface de predição (inferência)
# ─────────────────────────────────────────────
def traduzir_especie(codigo: int) -> str:
    """Tradução inversa: número → nome da espécie por extenso."""
    return SPECIES_MAP.get(int(codigo), "Espécie desconhecida")


def predizer_especie_interativo(modelo: DecisionTreeClassifier) -> None:
    """
    Lê dados arbitrários do usuário via input() e retorna a espécie
    prevista pelo modelo com o nome por extenso.
    """
    print("\n" + "═" * 50)
    print("  PASSO 9 — Classificar um novo pinguim")
    print("═" * 50)
    print("Informe os dados do pinguim (pressione Enter após cada valor):\n")

    try:
        island_input = input("  Ilha (0=Biscoe | 1=Dream | 2=Torgersen): ").strip()
        sex_input    = input("  Sexo (0=FEMALE | 1=MALE)               : ").strip()
        culmen_len   = input("  Comprimento do cúlmen (mm)              : ").strip()
        culmen_dep   = input("  Profundidade do cúlmen (mm)             : ").strip()
        flipper_len  = input("  Comprimento da nadadeira (mm)           : ").strip()
        body_mass    = input("  Massa corporal (g)                      : ").strip()

        amostra = pd.DataFrame([[
            int(island_input),
            int(sex_input),
            float(culmen_len),
            float(culmen_dep),
            float(flipper_len),
            float(body_mass),
        ]], columns=FEATURE_COLUMNS)

        codigo_previsto = modelo.predict(amostra)[0]
        especie = traduzir_especie(codigo_previsto)

        print("\n" + "─" * 50)
        print(f"  ✅ Espécie prevista: {especie} (código {codigo_previsto})")
        print("─" * 50 + "\n")

    except ValueError:
        print("\n  ❌ Erro: entrada inválida. Certifique-se de digitar apenas números.\n")


def predizer_especie_programatico(modelo: DecisionTreeClassifier, valores: list) -> str:
    """
    Versão sem input() para uso programático / testes automatizados.

    Parâmetros
    ----------
    modelo  : modelo treinado
    valores : lista com 6 elementos na ordem:
              [island, sex, culmen_length_mm, culmen_depth_mm,
               flipper_length_mm, body_mass_g]

    Retorno
    -------
    Nome da espécie por extenso ("Adelie", "Chinstrap" ou "Gentoo")

    Exemplo de uso
    --------------
    >>> especie = predizer_especie_programatico(modelo, [0, 1, 39.1, 18.7, 181, 3750])
    >>> print(especie)  # → "Adelie"
    """
    if len(valores) != 6:
        raise ValueError(f"Esperados 6 valores, recebidos {len(valores)}.")

    amostra = pd.DataFrame([valores], columns=FEATURE_COLUMNS)
    codigo  = modelo.predict(amostra)[0]
    return traduzir_especie(codigo)


# ─────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  AG2 — Classificador de Espécies de Pinguins")
    print("  Inatel | Engenharia de Computação e Software")
    print("=" * 55 + "\n")

   
    df = carregar_dados()
    print(f"[Dados] DataFrame carregado: {df.shape[0]} linhas × {df.shape[1]} colunas")
    print(df.to_string(index=False))
    print()

  
    X_train, X_test, y_train, y_test = dividir_dados(df)

   
    modelo = treinar_modelo(X_train, y_train)
    y_pred = predizer(modelo, X_test)

    
    exibir_metricas(y_test, y_pred, modelo, X_train, y_train)

   
    
    exemplo = [0, 1, 39.1, 18.7, 181, 3750]
    especie_exemplo = predizer_especie_programatico(modelo, exemplo)
    print(f"[Passo 9 – Exemplo] Entrada: {exemplo}")
    print(f"                    Espécie : {especie_exemplo}\n")

    # Interface interativa via terminal
    predizer_especie_interativo(modelo)
