"""CurTiPot Python - Web Application for Physical-Chemical Acid-Base Calculations.

Reimplementation of Ivano Gutz's CurTiPot suite in modern Python with Streamlit and Plotly.
"""

from __future__ import annotations

import streamlit as st

from curtipot.ui import (
    tab_analysis_fit,
    tab_analysis_spline,
    tab_database,
    tab_distribution,
    tab_ph,
    tab_simulator,
)


def apply_custom_styles() -> None:
    """Inject clean modern CSS for chemical application layout."""
    st.markdown(
        """
        <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 95%;
        }

        /* Metric cards custom styling */
        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.06);
            border-radius: 8px;
            padding: 10px 14px;
            border: 1px solid rgba(128, 128, 128, 0.15);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 8px 18px;
            border-radius: 6px 6px 0px 0px;
            font-weight: 500;
        }

        /* Sidebar credits and footer */
        .sidebar-footer {
            font-size: 0.82rem;
            color: #888888;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(128, 128, 128, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render the sidebar with documentation, credits, and navigation information."""
    with st.sidebar:
        st.title("🧪 CurTiPot Python")
        st.caption("Suíte de Equilíbrio Ácido-Base e Titulação Potenciométrica")

        st.markdown("---")
        st.markdown(
            """
            **Módulos Disponíveis:**
            - **1. pH & Tampão**: Cálculo de pH exato, força iônica e especiação.
            - **2. Especiação**: Diagramas de frações molares $\\alpha$, $\\log \\alpha$, $\\bar{n}_H$, $\\beta$ e $pI$.
            - **3. Simulador**: Simulação potenciométrica com diluição e derivadas.
            - **4. Análise I**: Interpolação por Spline Cúbica e detecção de $V_{eq}$.
            - **5. Análise II**: Regressão não-linear (RNLMP) para concentração e $pK_a$.
            - **6. Banco de Dados**: Consulta e exportação de 266 substâncias.
            """
        )

        st.markdown("---")
        st.markdown(
            """
            <div class="sidebar-footer">
                <b>Referência Científica:</b><br>
                Reimplementação em Python baseada no software <i>CurTiPot</i>
                criado pelo <b>Prof. Dr. Ivano G. R. Gutz</b> (Instituto de Química - USP).<br><br>
                <b>Tecnologias:</b> Python 3, NumPy, SciPy, Plotly, Streamlit.
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="CurTiPot Python",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()
    render_sidebar()

    # Application Header
    st.title("⚗️ CurTiPot: Equilíbrio Químico e Titulação")

    # Main tabs
    tab_titles = [
        "🧪 1. pH & Tampão",
        "📊 2. Especiação",
        "⚖️ 3. Simulador de Titulação",
        "📈 4. Análise I (Spline)",
        "🎯 5. Análise II (Regressão)",
        "📚 6. Banco de Constantes",
    ]

    t1, t2, t3, t4, t5, t6 = st.tabs(tab_titles)

    with t1:
        tab_ph.render()
    with t2:
        tab_distribution.render()
    with t3:
        tab_simulator.render()
    with t4:
        tab_analysis_spline.render()
    with t5:
        tab_analysis_fit.render()
    with t6:
        tab_database.render()


if __name__ == "__main__":
    main()
