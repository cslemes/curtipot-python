"""Streamlit UI tab for exploring the CurTiPot Chemical Database."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from curtipot.database import get_database_dataframe, get_system_by_name, normalize_text


def render() -> None:
    """Render the Database Explorer tab."""
    st.markdown("### 📚 Banco de Constantes Químicas CurTiPot")
    st.caption(
        "Consulta interativa ao banco de dados termodinâmicos com constantes de dissociação (pKₐ), "
        "cargas de espécies, massas molares e fórmulas químicas compiladas por Ivano Gutz."
    )

    try:
        df_all = get_database_dataframe()
    except Exception as e:
        st.error(f"Erro ao carregar banco de dados: {e}")
        return

    # Filter section
    col_search, col_cat = st.columns([2, 1])

    with col_search:
        search_query = st.text_input(
            "🔍 Pesquisar por nome ou fórmula química:",
            placeholder="ex: acetico, fosforico, edta, h2so4...",
        )

    categories = ["Todas"] + sorted([str(c) for c in df_all["category"].unique() if str(c).strip()])
    with col_cat:
        selected_cat = st.selectbox("Categoria:", options=categories)

    # Apply filters
    filtered_df = df_all.copy()

    if selected_cat != "Todas":
        filtered_df = filtered_df[filtered_df["category"] == selected_cat]

    if search_query.strip():
        norm_q = normalize_text(search_query)
        def matches(row: pd.Series) -> bool:
            name_m = norm_q in normalize_text(str(row["name"]))
            form_m = norm_q in normalize_text(str(row.get("formula", "")))
            return name_m or form_m

        filtered_df = filtered_df[filtered_df.apply(matches, axis=1)]

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total de Substâncias", len(df_all))
    m2.metric("Substâncias Filtradas", len(filtered_df))
    m3.metric("Categorias Disponíveis", len(categories) - 1)

    # Table view
    st.markdown("##### 📋 Tabela de Substâncias")
    display_cols = [
        "name",
        "category",
        "charge_base",
        "pKa1",
        "pKa2",
        "pKa3",
        "pKa4",
        "formula",
        "molar_mass",
        "temp_c",
    ]
    cols_to_show = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[cols_to_show].style.format({
            "pKa1": "{:.3f}",
            "pKa2": "{:.3f}",
            "pKa3": "{:.3f}",
            "pKa4": "{:.3f}",
            "molar_mass": "{:.2f}",
        }, na_rep="-"),
        width="stretch",
        height=380,
    )

    # Detailed Substance Inspector
    if not filtered_df.empty:
        st.markdown("---")
        st.markdown("##### 🔬 Detalhes da Substância Selecionada")

        substance_names = list(filtered_df["name"].unique())
        selected_sub_name = st.selectbox(
            "Selecione uma substância para inspeção detalhada:",
            options=substance_names,
            key="db_detail_select",
        )

        try:
            sys = get_system_by_name(selected_sub_name)

            d1, d2, d3, d4 = st.columns(4)
            d1.write(f"**Fórmula:** `{sys.formula or 'N/A'}`")
            d2.write(f"**Massa Molar:** `{sys.molar_mass:.2f} g/mol`" if sys.molar_mass else "**Massa Molar:** `N/A`")
            d3.write(f"**Carga Base ($z_B$):** `{sys.charge_base}`")
            d4.write(f"**Prótons Dissociáveis:** `{sys.num_protons}`")

            # Protonation steps breakdown
            if sys.num_protons > 0:
                pka_details = []
                betas = sys.betas
                log_betas = sys.log_betas
                for i in range(sys.num_protons):
                    pka_val = sys.pKas[i]
                    pka_details.append({
                        "Etapa": f"pKₐ{i+1}",
                        "pKₐ": f"{pka_val:.4f}",
                        "Kₐ": f"{10.0**(-pka_val):.3e}",
                        "log(β)": f"{log_betas[i+1]:.4f}",
                        "β": f"{betas[i+1]:.3e}",
                    })
                st.dataframe(pd.DataFrame(pka_details), width="stretch")

            # Species labels
            st.write("**Espécies e Cargas no Equilíbrio:**")
            species_info = []
            for lab, ch in zip(sys.species_names, sys.species_charges):
                species_info.append(f"**{lab}** (Carga: `{ch:+d}`)")
            st.markdown(" &bull; ".join(species_info))

        except Exception as e:
            st.error(f"Erro ao carregar detalhes: {e}")

    # Export
    st.markdown("---")
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Baixar Tabela Filtrada em CSV",
        data=csv_bytes,
        file_name="banco_constantes_curtipot.csv",
        mime="text/csv",
    )


render_tab_database = render
