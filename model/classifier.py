import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

SPECIES_MAP = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

FEATURE_COLUMNS = [
    "island", "culmen_length_mm", "culmen_depth_mm",
    "flipper_length_mm", "body_mass_g", "sex",
]
TARGET_COLUMN = "species"


def carregar_dados() -> pd.DataFrame:
    base = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base, "palmerpenguins_final.csv"))


def dividir_dados(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[Passo 5] Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras\n")
    return X_train, X_test, y_train, y_test


def treinar_modelo(X_train, y_train) -> DecisionTreeClassifier:
    modelo = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)
    print("[Passos 6-7] Modelo treinado.\n")
    return modelo


def exibir_metricas(y_test, y_pred, modelo) -> None:
    nomes = list(SPECIES_MAP.values())

    print("[Passo 8] Classification Report:")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=nomes, zero_division=0))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("AG2 Inatel — Avaliação do Classificador de Pinguins", fontweight="bold")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    ConfusionMatrixDisplay(cm, display_labels=nomes).plot(cmap="Blues", ax=axes[0])
    axes[0].set_title("Matriz de Confusão (contagens)")

    cm_norm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2], normalize="true")
    ConfusionMatrixDisplay(cm_norm, display_labels=nomes).plot(cmap="Greens", ax=axes[1])
    axes[1].set_title("Matriz de Confusão (normalizada)")

    plot_tree(modelo, feature_names=FEATURE_COLUMNS, class_names=nomes, filled=True, rounded=True, ax=axes[2])
    axes[2].set_title("Estrutura de Decisão")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


def predizer_especie(modelo: DecisionTreeClassifier, valores: list) -> str:
    if len(valores) != 6:
        raise ValueError(f"Esperados 6 valores, recebidos {len(valores)}.")
    amostra = pd.DataFrame([valores], columns=FEATURE_COLUMNS)
    return SPECIES_MAP[modelo.predict(amostra)[0]]


def interface_predicao(modelo: DecisionTreeClassifier) -> None:
    print("\n" + "=" * 45)
    print("  PASSO 9 — Classificar um novo pinguim")
    print("=" * 45)

    try:
        entradas = [
            ("Ilha (0=Torgersen | 1=Biscoe | 2=Dream)", int),
            ("Sexo (0=MALE | 1=FEMALE)", int),
            ("Comprimento do cúlmen (mm)", float),
            ("Profundidade do cúlmen (mm)", float),
            ("Comprimento da nadadeira (mm)", float),
            ("Massa corporal (g)", float),
        ]
        valores = [tipo(input(f"  {label}: ").strip()) for label, tipo in entradas]
        especie = predizer_especie(modelo, valores)
        print(f"\n  Espécie prevista: {especie}\n")

    except ValueError:
        print("\n  Entrada inválida. Digite apenas números.\n")


if __name__ == "__main__":
    df = carregar_dados()
    print(f"[Dados] {df.shape[0]} linhas × {df.shape[1]} colunas\n")

    X_train, X_test, y_train, y_test = dividir_dados(df)

    modelo = treinar_modelo(X_train, y_train)
    y_pred = modelo.predict(X_test)

    exibir_metricas(y_test, y_pred, modelo)

    # Exemplo programático do Passo 9
    exemplo = [0, 0, 39.1, 18.7, 181, 3750]
    print(f"[Passo 9] Entrada: {exemplo} → {predizer_especie(modelo, exemplo)}\n")

    interface_predicao(modelo)
