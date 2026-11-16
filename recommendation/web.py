"""Aplicação web (Streamlit) do framework de recomendação.

    .venv/bin/streamlit run recommendation/web.py

Mesmo motor da CLI (`recommendation.engine`). Entradas: volume, consulta
(catálogo ou EXPLAIN colado), envelope do cluster e objetivo. Saídas:
recomendação com faixa de erro, fronteira de Pareto, cascata SHAP e, quando a
célula foi medida, a melhor célula real para conferência.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from recommendation.engine import (ClusterEnvelope, explain, load_catalog, query_from_catalog,  # noqa: E402
                                   query_from_explain, query_from_sql, recommend, regret_if_measured, sf_equivalent)
from recommendation.engine.recommender import ERROR_MEDIAN, MEASURED_SFS, TIE_TOLERANCE, WITHIN_1_5X  # noqa: E402

ROTULOS = {
    "scale_factor": "Fator de escala", "workers": "Workers", "executor_cores": "Núcleos/executor",
    "executor_memory_gb": "Memória/executor (GiB)", "total_cores": "Núcleos totais",
    "total_memory_gb": "Memória total (GiB)", "cores_per_gb_data": "Núcleos por GiB de dados",
    "memory_per_gb_data": "Memória por GiB de dados", "data_per_worker_gb": "Dados por worker (GiB)",
    "query_category_enc": "Categoria", "num_joins": "Junções", "num_broadcast_joins": "Junções broadcast",
    "num_sort_merge_joins": "Junções sort-merge", "num_exchanges": "Trocas", "num_sorts": "Ordenações",
    "num_aggregates": "Agregações", "num_filters": "Filtros", "num_scans": "Varreduras",
    "num_subqueries": "Subconsultas", "num_windows": "Janelas", "num_unions": "Uniões",
    "plan_depth": "Profundidade do plano", "plan_lines": "Linhas do plano", "plan_complexity": "Complexidade do plano",
}

st.set_page_config(page_title="spark-recommend", layout="wide")
st.title("Recomendação de configuração Spark sobre Kubernetes")
st.caption("Modelo treinado sobre 75 células / 22.218 execuções TPC-DS no cluster da RNP. "
           f"Erro mediano da previsão de tempo {ERROR_MEDIAN:.0%}; {WITHIN_1_5X:.0%} das previsões dentro de 1,5×.")

cat = load_catalog()

with st.sidebar:
    st.header("Carga")
    modo_vol = st.radio("Volume", ["Fator de escala nominal", "Tamanho da Silver (GiB)"], horizontal=True)
    if modo_vol.startswith("Fator"):
        sf = float(st.select_slider("SF", options=MEASURED_SFS, value=100))
        gib = None
    else:
        gib = float(st.number_input("GiB", min_value=1.0, max_value=2000.0, value=120.0, step=10.0))
        sf = sf_equivalent(gib)
        st.caption(f"≈ SF {sf:.0f}")
    modo_q = st.radio("Consulta", ["Do catálogo (TPC-DS)", "SQL escrita por mim", "EXPLAIN EXTENDED colado"])
    if modo_q.startswith("Do"):
        qname = st.selectbox("Consulta", cat["queries"], index=cat["queries"].index("q26") if "q26" in cat["queries"] else 0)
        query = query_from_catalog(qname, sf, cat)
    elif modo_q.startswith("SQL"):
        qname = st.text_input("Nome", "minha_consulta")
        sql = st.text_area("SQL sobre o esquema TPC-DS", height=200,
                           placeholder="SELECT i_item_desc, SUM(ss_quantity) FROM store_sales JOIN item ON ss_item_sk = i_item_sk GROUP BY 1 LIMIT 100")
        cat_opt = st.selectbox("Categoria estrutural", ["(inferir do plano)", "A_scan", "B_shuffle", "C_subquery"])
        if sql.strip():
            with st.spinner("Gerando o plano físico numa sessão Spark local (primeira vez ~20 s)..."):
                try:
                    query = query_from_sql(qname, sql, sf, None if cat_opt.startswith("(") else cat_opt)
                except Exception as ex:
                    st.error(f"Não consegui planejar a consulta: {str(ex)[:400]}")
                    query = None
        else:
            query = None
    else:
        qname = st.text_input("Nome", "minha_consulta")
        categoria = st.selectbox("Categoria estrutural", ["A_scan", "B_shuffle", "C_subquery"], index=1)
        plano = st.text_area("Saída de EXPLAIN EXTENDED", height=160)
        query = query_from_explain(qname, plano, categoria) if plano.strip() else None

    st.header("Cluster")
    nodes = st.text_input("Nós (NxCORESc MEMg, vírgula)", "5x48c187g,2x24c172g")
    env = ClusterEnvelope.parse(nodes)
    st.header("Objetivo")
    objective = st.radio("Otimizar", ["time", "cost", "usd"], format_func=lambda o: {"time": "menor tempo", "cost": "menor core-hora (até 1,5× do mais rápido)", "usd": "menor custo em dólar"}[o])
    top_k = st.slider("Candidatas no topo", 1, 10, 3)

if query is None:
    st.info("Escolha uma consulta do catálogo, escreva uma SQL ou cole um EXPLAIN EXTENDED.")
    st.stop()

rec = recommend(query, sf, env, objective=objective, top_k=top_k, data_gib=gib, catalog=cat)
best = rec.best

c1, c2, c3 = st.columns(3)
c1.metric("Recomendação", f"{best.profile} × W{int(best.workers)}", f"{int(best.total_cores)} núcleos")
c2.metric("Tempo previsto", f"{best.pred_s:.0f} s", f"{best.pred_lo_s:.0f}–{best.pred_hi_s:.0f} s (1,5×)")
c3.metric("Custo previsto", f"{best.core_hours:.2f} core-h", f"US$ {best.cost_usd:.2f}")
for w in rec.warnings:
    st.warning(w)

st.subheader(f"Top {top_k}")
top = rec.top.copy()
top["empate"] = top.tie_with_best.map({True: f"≤{TIE_TOLERANCE:.0%} do melhor", False: ""})
st.dataframe(top[["profile", "workers", "total_cores", "pred_s", "core_hours", "cost_usd", "pareto", "measured_cell", "empate"]]
             .rename(columns={"profile": "perfil", "pred_s": "tempo (s)", "core_hours": "core-h", "cost_usd": "US$",
                              "pareto": "fronteira", "measured_cell": "célula medida"}),
             hide_index=True, use_container_width=True)

r = regret_if_measured(rec)
if rec.measured is not None:
    m = rec.measured
    st.subheader("Conferência com a matriz medida")
    txt = f"Melhor célula medida para {query.name} em SF {int(m.scale_factor.iloc[0])}: **{m.iloc[0].profile} × W{int(m.iloc[0].workers)}** ({m.iloc[0].measured_s:.1f} s)."
    if r is not None:
        txt += f" A recomendação foi medida em {m[(m.profile == best.profile) & (m.workers == best.workers)].measured_s.iloc[0]:.1f} s, regret de **{100*r:.1f}%**."
    st.markdown(txt)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Fronteira de Pareto (previsão)")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    cands = rec.candidates
    for prof, g in cands.groupby("profile"):
        ax.scatter(g.pred_s, g.core_hours, s=55, label=prof, edgecolor="white", linewidth=0.5, zorder=3)
        for _, row in g.iterrows():
            ax.annotate(f"W{int(row.workers)}", (row.pred_s, row.core_hours), textcoords="offset points", xytext=(5, -8), fontsize=7, color="#555")
    f = cands[cands.pareto].sort_values("pred_s")
    ax.plot(f.pred_s, f.core_hours, "--", color="#333", linewidth=1.1, zorder=2)
    ax.scatter([best.pred_s], [best.core_hours], s=320, marker="*", color="#c1121f", edgecolor="white", zorder=5, label="recomendação")
    ax.set_xlabel("Tempo previsto (s)"); ax.set_ylabel("Custo previsto (core-hours)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, title="perfil", title_fontsize=8)
    st.pyplot(fig, use_container_width=True)
with col_b:
    st.subheader("Por que essa recomendação (SHAP)")
    try:
        import shap
        e = explain(query, best)
        e.feature_names = [ROTULOS.get(n, n) for n in e.feature_names]
        plt.figure(figsize=(6, 4.2))
        shap.plots.waterfall(e, max_display=10, show=False)
        st.pyplot(plt.gcf(), use_container_width=True)
        st.caption("Contribuições em log(1+ms); features de capacidade e de escala são colineares entre si, leia-as como grupo.")
    except Exception as ex:  # SHAP é opcional na interface
        st.info(f"SHAP indisponível: {ex}")

with st.expander("Todas as candidatas viáveis"):
    st.dataframe(cands[["profile", "workers", "total_cores", "total_memory_gb", "pred_s", "core_hours", "cost_usd", "pareto", "measured_cell"]],
                 hide_index=True, use_container_width=True)
with st.expander("Descrição da consulta usada pelo modelo"):
    st.json({"nome": query.name, "categoria": query.category, "origem": query.source, **{ROTULOS.get(k, k): v for k, v in query.plan.items()}})
    if getattr(query, "plan_text", None):
        st.code(query.plan_text[query.plan_text.find("== Physical Plan =="):][:6000], language="text")
