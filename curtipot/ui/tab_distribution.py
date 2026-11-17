"""Streamlit UI tab for chemical speciation distribution diagrams."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from curtipot.database import get_system_by_name, list_substance_names
from curtipot.distribution import (
    calculate_distribution,
    find_isoelectric_point,
)
from curtipot.models import ChemicalSystem


def render() -> None:
    """Render the Speciation Distribution tab."""
    st.markdown("### 📊 Diagramas de Especiação e Equilíbrio")
    st.caption(
        "Gera diagramas de distribuição de espécies (α vs pH, log(α)), "
        "curva de formação (n̄_H), capacidade tamponante (β) e cálculo do ponto isoelétrico (pI)."
    )

    all_names = list_substance_names()

    col_sub, col_params = st.columns([2, 1])

    with col_sub:
        use_custom = st.checkbox("Substância personalizada", key="dist_custom")
        if not use_custom:
            default_idx = 0
            if "Ácido fosfórico" in all_names:
                default_idx = all_names.index("Ácido fosfórico")
            sub_name = st.selectbox(
                "Substância",
                options=all_names,
                index=default_idx,
                key="dist_sub_select",
            )
            try:
                system = get_system_by_name(sub_name, concentration=0.10)
            except Exception as e:
                st.error(f"Erro ao carregar substância: {e}")
                return
        else:
            sub_name = st.text_input("Nome da Substância", value="Ácido Poliprótico", key="dist_cname")
            charge_b = st.number_input("Carga da Base Desprotonada", value=-3, step=1, key="dist_cz")
            pkas_raw = st.text_input("pKas (separados por vírgula)", value="2.148, 7.198, 12.375", key="dist_cpka")
            try:
                pkas = [float(k.strip()) for k in pkas_raw.split(",") if k.strip()]
                system = ChemicalSystem(
                    name=sub_name,
                    charge_base=int(charge_b),
                    pKas=pkas,
                    concentration=0.10,
                )
            except Exception as e:
                st.error(f"Definição inválida da substância: {e}")
                return

    with col_params:
        conc = st.number_input(
            "Concentração Total [mol/L]",
            min_value=1e-5,
            max_value=10.0,
            value=0.10,
            step=0.01,
            format="%.4f",
            key="dist_conc",
        )
        system = system.with_concentration(conc)

        ph_min, ph_max = st.slider(
            "Faixa de pH",
            min_value=0.0,
            max_value=14.0,
            value=(0.0, 14.0),
            step=0.5,
            key="dist_ph_range",
        )
        resolution = st.slider(
            "Pontos de cálculo",
            min_value=71,
            max_value=501,
            value=141,
            step=10,
            key="dist_resolution",
        )

    # Compute distribution
    ph_array = np.linspace(ph_min, ph_max, resolution, dtype=float)
    try:
        res = calculate_distribution(system, ph_values=ph_array)
    except Exception as e:
        st.error(f"Erro no cálculo de distribuição: {e}")
        return

    # Check for isoelectric point
    pi_val = find_isoelectric_point(res)

    st.markdown("---")
    c_m1, c_m2, c_m3 = st.columns(3)
    c_m1.metric("Número de Prótons Dissociáveis", f"{system.num_protons}")
    c_m2.metric(
        "Constantes pKa",
        ", ".join([f"{k:.2f}" for k in system.pKas]) if system.pKas else "Nenhum (Eletrólito Forte)",
    )
    if pi_val is not None:
        c_m3.metric("Ponto Isoelétrico (pI)", f"{pi_val:.3f}")
    else:
        c_m3.metric("Ponto Isoelétrico (pI)", "N/A (sem inversão de carga)")

    # Diagrams tabs
    tab_alpha, tab_log, tab_formation, tab_buffer, tab_data = st.tabs(
        [
            "Diagrama α vs pH",
            "Diagrama log(α) vs pH",
            "Função de Formação (n̄_H)",
            "Capacidade Tampão (β)",
            "Tabela de Dados",
        ]
    )

    with tab_alpha:
        fig_alpha = go.Figure()
        for i, label in enumerate(res.species_labels):
            fig_alpha.add_trace(
                go.Scatter(
                    x=res.ph_array,
                    y=res.alphas[i, :],
                    mode="lines",
                    name=f"α({label})",
                    line=dict(width=2.5),
                    hovertemplate=f"<b>{label}</b><br>pH: %{{x:.2f}}<br>α: %{{y:.4f}}<extra></extra>",
                )
            )
        # Add vertical dashed lines for pKas
        for k_idx, pka_val in enumerate(system.pKas):
            if ph_min <= pka_val <= ph_max:
                fig_alpha.add_vline(
                    x=pka_val,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text=f"pKₐ{k_idx+1}={pka_val:.2f}",
                    annotation_position="top",
                )
        fig_alpha.update_layout(
            title=f"Distribuição de Espécies: {system.name}",
            xaxis_title="pH",
            yaxis_title="Fração Molar (α)",
            yaxis=dict(range=[-0.02, 1.05]),
            template="plotly_white",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_alpha, width="stretch")

    with tab_log:
        fig_log = go.Figure()
        for i, label in enumerate(res.species_labels):
            fig_log.add_trace(
                go.Scatter(
                    x=res.ph_array,
                    y=res.log_alphas[i, :],
                    mode="lines",
                    name=f"log α({label})",
                    line=dict(width=2.2),
                    hovertemplate=f"<b>{label}</b><br>pH: %{{x:.2f}}<br>log α: %{{y:.2f}}<extra></extra>",
                )
            )
        fig_log.update_layout(
            title=f"Diagrama Logarítmico de Especiação: {system.name}",
            xaxis_title="pH",
            yaxis_title="log₁₀(α)",
            yaxis=dict(range=[-10.0, 0.2]),
            template="plotly_white",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_log, width="stretch")

    with tab_formation:
        fig_form = go.Figure()
        fig_form.add_trace(
            go.Scatter(
                x=res.ph_array,
                y=res.n_bar,
                mode="lines",
                name="n̄_H (Número médio de prótons)",
                line=dict(color="#1f77b4", width=2.5),
                hovertemplate="pH: %{x:.2f}<br>n̄_H: %{y:.3f}<extra></extra>",
            )
        )
        fig_form.add_trace(
            go.Scatter(
                x=res.ph_array,
                y=res.average_charge,
                mode="lines",
                name="Carga Média da Espécie",
                line=dict(color="#d62728", width=2.5, dash="dash"),
                hovertemplate="pH: %{x:.2f}<br>Carga: %{y:.3f}<extra></extra>",
            )
        )
        if pi_val is not None and ph_min <= pi_val <= ph_max:
            fig_form.add_vline(
                x=pi_val,
                line_dash="dashdot",
                line_color="green",
                annotation_text=f"pI = {pi_val:.3f}",
                annotation_position="bottom right",
            )
        fig_form.update_layout(
            title=f"Curvas de Formação e Carga Média: {system.name}",
            xaxis_title="pH",
            yaxis_title="Valor Médio",
            template="plotly_white",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_form, width="stretch")

    with tab_buffer:
        fig_buf = go.Figure()
        fig_buf.add_trace(
            go.Scatter(
                x=res.ph_array,
                y=res.buffer_capacity,
                mode="lines",
                name=f"β (C = {conc:.3f} M)",
                line=dict(color="#2ca02c", width=2.5),
                fill="tozeroy",
                hovertemplate="pH: %{x:.2f}<br>β: %{y:.4f} mol/(L·pH)<extra></extra>",
            )
        )
        for k_idx, pka_val in enumerate(system.pKas):
            if ph_min <= pka_val <= ph_max:
                fig_buf.add_vline(
                    x=pka_val,
                    line_dash="dot",
                    line_color="gray",
                    annotation_text=f"pKₐ{k_idx+1}={pka_val:.2f}",
                )
        fig_buf.update_layout(
            title=f"Capacidade de Tamponamento (β vs pH): {system.name}",
            xaxis_title="pH",
            yaxis_title="Capacidade Tamponante β [mol / (L · pH)]",
            template="plotly_white",
            height=450,
        )
        st.plotly_chart(fig_buf, width="stretch")

    with tab_data:
        data_dict: dict[str, object] = {
            "pH": res.ph_array,
            "n_bar": res.n_bar,
            "Carga Média": res.average_charge,
            "Cap. Tampão (β)": res.buffer_capacity,
        }
        for i, label in enumerate(res.species_labels):
            data_dict[f"alpha_{label}"] = res.alphas[i, :]
        df_dist = pd.DataFrame(data_dict)
        st.dataframe(df_dist, width="stretch")
        csv_data = df_dist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Baixar Dados da Especiação (CSV)",
            data=csv_data,
            file_name=f"especiação_{system.name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )


render_tab_distribution = render
