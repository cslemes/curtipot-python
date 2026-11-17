# CurTiPot Python

Reimplementação em Python moderno da suíte físico-química **CurTiPot** (*Programa de Cálculo de pH e Simulação e Análise de Curvas de Titulação*), originalmente desenvolvida em Excel/VBA pelo **Prof. Dr. Ivano G. R. Gutz** (Instituto de Química - Universidade de São Paulo, IQ-USP).

---

## 🚀 Gerenciamento do Ambiente com `uv`

Este projeto é 100% gerenciado via [uv](https://docs.astral.sh/uv/), garantindo resolução determinística de dependências, lockfile padronizado (`uv.lock`) e execução ultrarrápida.

### 1. Sincronizar o Ambiente e Dependências

Para criar o ambiente virtual e instalar todas as dependências (incluindo o grupo de desenvolvimento `dev` com pytest):

```bash
uv sync
```

### 2. Executar a Interface Web (Streamlit)

Para iniciar a aplicação interativa do CurTiPot:

```bash
uv run streamlit run curtipot/app.py
```

O aplicativo abrirá automaticamente no seu navegador em `http://localhost:8501`.

### 3. Executar os Testes Unitários

Para rodar toda a suíte de testes (62 testes com conformidade química estrita):

```bash
uv run pytest -v
```

---

## ☁️ Deploy no Streamlit Community Cloud

O projeto está pronto para deploy com 1 clique no [Streamlit Community Cloud](https://share.streamlit.io):

1. Acesse o **Streamlit Community Cloud** e clique em **"Create app"**.
2. Conecte sua conta do GitHub e preencha os campos:
   - **Repository:** `cslemes/curtipot-python`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py` *(ou `curtipot/app.py`)*
3. Clique em **"Deploy!"**. As dependências de produção são lidas automaticamente do `requirements.txt`.

---

## 🧪 Estrutura dos Módulos

- **`curtipot.models`**: Dataclasses de domínio (`ChemicalSystem`, `TitrationConfig`, `EquilibriumResult`, etc.).
- **`curtipot.database`**: Carregador e busca indexada com as 263 constantes químicas originais.
- **`curtipot.equilibrium`**: Balanço de prótons e correções de atividade pela Equação de Davies ($|\Delta\text{pH}| \le 10^{-4}$).
- **`curtipot.distribution`**: Diagramas de especiação ($\alpha_i$, $\log \alpha$), função de formação $\bar{n}_H$, carga média e capacidade tamponante $\beta$.
- **`curtipot.titration`**: Simulação potenciométrica com diluição contínua de amostra e titulante, derivadas numéricas ($dpH/dV$, $d^2pH/dV^2$) e equivalência.
- **`curtipot.analysis`**:
  - **Análise I**: Interpolação por *Natural Cubic Splines* e refinamento de passagens por zero.
  - **Análise II**: Regressão não-linear por mínimos quadrados (`least_squares`) para estimar concentrações e $pK_a$.
- **`curtipot.app`**: Interface visual em Streamlit com 6 abas e gráficos interativos Plotly.
