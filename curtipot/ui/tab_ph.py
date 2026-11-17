"""Streamlit UI tab for pH and Buffer calculations."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from curtipot.database import get_system_by_name, list_substance_names
from curtipot.equilibrium import solve_ph
from curtipot.models import ChemicalSystem


def render() -> None:
    """Render the pH & Buffer calculation tab."""
    st.markdown("### 🧪 Cálculo de pH e Soluções Tampão")
    st.caption(
        "Calcula o pH exato, especiação de equilíbrio, força iônica e capacidade de tamponamento "
        "com modelo de Davies para correções de atividade."
    )

    all_names = list_substance_names()

    col_conf1, col_conf2 = st.columns([2, 1])
    with col_conf2:
        calc_activity = st.checkbox(
            "Correção de Davies (Atividade)",
            value=True,
            help="Aplica modelo de Davies iterativo para coeficientes de atividade.",
        )
        temp_c = st.number_input(
            "Temperatura (°C)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help="Temperatura da solução em Celsius.",
        )
        inert_salt_conc = st.number_input(
            "Sal inerte (ex: NaCl) [mol/L]",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.01,
            format="%.4f",
            help="Eletrólito inerte 1:1 adicionado (ex: NaCl, KNO3).",
        )

    with col_conf1:
        num_systems = st.radio(
            "Composição do Sistema",
            options=["Componente Único", "Mistura Binária (Tampão / 2 Componentes)"],
            horizontal=True,
        )

        systems: list[ChemicalSystem] = []

        # Component 1
        st.markdown("#### Componente 1")
        c1_sub, c1_conc = st.columns([3, 2])
        with c1_sub:
            use_custom_1 = st.checkbox("Substância personalizada", key="ph_custom_1")
            if not use_custom_1:
                default_idx = 0
                if "Ácido acético" in all_names:
                    default_idx = all_names.index("Ácido acético")
                sub1_name = st.selectbox(
                    "Substância 1",
                    options=all_names,
                    index=default_idx,
                    key="ph_sub1_select",
                )
            else:
                sub1_name = st.text_input("Nome", value="Ácido Customizado 1", key="ph_custom_name_1")
                sub1_charge = st.number_input("Carga da Base Desprotonada", value=-1, step=1, key="ph_custom_z_1")
                sub1_pka_str = st.text_input("pKas (separados por vírgula)", value="4.757", key="ph_custom_pka_1")

        with c1_conc:
            conc1 = st.number_input(
                "Concentração [mol/L]",
                min_value=0.0,
                max_value=10.0,
                value=0.100,
                step=0.01,
                format="%.5f",
                key="ph_conc_1",
            )

        if not use_custom_1:
            try:
                sys1 = get_system_by_name(sub1_name, concentration=conc1)
                systems.append(sys1)
            except Exception as e:
                st.error(f"Erro ao carregar substância 1: {e}")
        else:
            try:
                pkas_1 = [float(k.strip()) for k in sub1_pka_str.split(",") if k.strip()]
                sys1 = ChemicalSystem(
                    name=sub1_name,
                    charge_base=int(sub1_charge),
                    pKas=pkas_1,
                    concentration=conc1,
                )
                systems.append(sys1)
            except Exception as e:
                st.error(f"Erro na definição da substância 1: {e}")

        # Component 2 (if selected)
        if num_systems.startswith("Mistura"):
            st.markdown("#### Componente 2")
            c2_sub, c2_conc = st.columns([3, 2])
            with c2_sub:
                use_custom_2 = st.checkbox("Substância personalizada", key="ph_custom_2")
                if not use_custom_2:
                    default_idx2 = 0
                    if "Acetato de sódio" in all_names:
                        default_idx2 = all_names.index("Acetato de sódio")
                    elif "Hidróxido de sódio" in all_names:
                        default_idx2 = all_names.index("Hidróxido de sódio")
                    sub2_name = st.selectbox(
                        "Substância 2",
                        options=all_names,
                        index=default_idx2,
                        key="ph_sub2_select",
                    )
                else:
                    sub2_name = st.text_input("Nome", value="Base Customizada 2", key="ph_custom_name_2")
                    sub2_charge = st.number_input("Carga da Base Desprotonada", value=-1, step=1, key="ph_custom_z_2")
                    sub2_pka_str = st.text_input("pKas (separados por vírgula)", value="4.757", key="ph_custom_pka_2")

            with c2_conc:
                conc2 = st.number_input(
                    "Concentração [mol/L]",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.100,
                    step=0.01,
                    format="%.5f",
                    key="ph_conc_2",
                )

            if not use_custom_2:
                try:
                    sys2 = get_system_by_name(sub2_name, concentration=conc2)
                    systems.append(sys2)
                except Exception as e:
                    st.error(f"Erro ao carregar substância 2: {e}")
            else:
                try:
                    pkas_2 = [float(k.strip()) for k in sub2_pka_str.split(",") if k.strip()]
                    sys2 = ChemicalSystem(
                        name=sub2_name,
                        charge_base=int(sub2_charge),
                        pKas=pkas_2,
                        concentration=conc2,
                    )
                    systems.append(sys2)
                except Exception as e:
                    st.error(f"Erro na definição da substância 2: {e}")

    # Calculation Execution
    if not systems:
        st.warning("Adicione ao menos uma substância válida para calcular.")
        return

    inert_ions = [(inert_salt_conc, 1), (inert_salt_conc, -1)] if inert_salt_conc > 0 else None

    try:
        res = solve_ph(
            systems=systems,
            inert_ions=inert_ions,
            calc_activity=calc_activity,
            temp_c=temp_c,
        )
    except Exception as e:
        st.error(f"Erro durante o cálculo de equilíbrio: {e}")
        return

    st.markdown("---")
    st.subheader("📊 Resultados de Equilíbrio")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("pH", f"{res.ph:.4f}")
    m2.metric("pOH", f"{res.poh:.4f}")
    m3.metric("[H⁺]", f"{res.h_conc:.3e} M")
    m4.metric("[OH⁻]", f"{res.oh_conc:.3e} M")
    m5.metric("Força Iônica (I)", f"{res.ionic_strength:.4f} M")
    m6.metric("Cap. Tampão (β)", f"{res.buffer_capacity:.4f}")

    col_meta1, col_meta2 = st.columns([1, 1])
    with col_meta1:
        st.info(
            f"**Coeficiente de atividade γ(H⁺):** `{res.activity_coeff_h:.4f}` | "
            f"**Convergência:** `{'Sim' if res.converged else 'Não'}` ({res.iterations} iterações)"
        )

    # Speciation Table and Chart
    st.markdown("#### 🧬 Especiação Química em Equilíbrio")

    spec_rows: list[dict[str, object]] = []
    for sys in systems:
        concs = res.species_concentrations.get(sys.name, [])
        labels = sys.species_names
        charges = sys.species_charges
        total_c = max(1e-15, sys.concentration)
        for i, (lab, ch, conc) in enumerate(zip(labels, charges, concs)):
            pct = (conc / total_c) * 100.0 if total_c > 0 else 0.0
            ch_str = f"{abs(ch)}-" if ch < -1 else ("-" if ch == -1 else (f"{ch}+" if ch > 1 else ("+" if ch == 1 else "0")))
            spec_rows.append({
                "Sistema": sys.name,
                "Espécie": lab,
                "Carga": ch_str,
                "Concentração (M)": f"{conc:.4e}",
                "Fração (%)": pct,
            })

    if spec_rows:
        df_spec = pd.DataFrame(spec_rows)
        tab_tbl, tab_chart = st.tabs(["Tabela de Espécies", "Gráfico de Especiação"])
        with tab_tbl:
            st.dataframe(
                df_spec.style.format({"Fração (%)": "{:.2f}%"}),
                use_container_width=True,
            )

        with tab_chart:
            fig = go.Figure()
            for sys_name in df_spec["Sistema"].unique():
                sub_df = df_spec[df_spec["Sistema"] == sys_name]
                fig.add_trace(
                    go.Bar(
                        x=sub_df["Espécie"],
                        y=sub_df["Fração (%)"],
                        name=sys_name,
                        text=[f"{v:.1f}%" for v in sub_df["Fração (%)"]],
                        textposition="auto",
                    )
                )
            fig.update_layout(
                title="Distribuição Percentual das Espécies em Equilíbrio",
                xaxis_title="Espécie Química",
                yaxis_title="Fração (%)",
                yaxis=dict(range=[0, 105]),
                template="plotly_white",
                height=380,
                margin=dict(l=40, r=40, t=50, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


render_tab_ph = render
