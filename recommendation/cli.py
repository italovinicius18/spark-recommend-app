"""spark-recommend: interface de linha de comando do framework.

    spark-recommend recommend --sf 500 --query q26 --nodes 5x48c187g,2x24c172g
    spark-recommend recommend --gib 120 --explain plano.txt --objective cost --json
    spark-recommend catalog
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from recommendation.engine import (ClusterEnvelope, load_catalog, query_from_catalog, query_from_explain,
                                   query_from_sql, recommend, regret_if_measured, sf_equivalent)

app = typer.Typer(add_completion=False, help=__doc__)


@app.command("recommend")
def cmd_recommend(
    query: Optional[str] = typer.Option(None, help="consulta do catálogo (q01..q99)"),
    explain_file: Optional[Path] = typer.Option(None, "--explain", help="arquivo com EXPLAIN EXTENDED"),
    sql_file: Optional[Path] = typer.Option(None, "--sql", help="arquivo .sql; o plano é gerado localmente na magnitude informada"),
    sf: Optional[float] = typer.Option(None, help="fator de escala nominal"),
    gib: Optional[float] = typer.Option(None, help="tamanho da Silver em GiB (alternativa a --sf)"),
    nodes: str = typer.Option("5x48c187g,2x24c172g", help="envelope: NxCORESc MEMg, separado por vírgula"),
    objective: str = typer.Option("time", help="time | cost | usd"),
    top: int = typer.Option(3),
    category: str = typer.Option("B_shuffle", help="categoria da consulta quando vem de --explain"),
    as_json: bool = typer.Option(False, "--json"),
):
    if sf is None and gib is None:
        raise typer.BadParameter("informe --sf ou --gib")
    sf_eff = sf if sf is not None else sf_equivalent(gib)
    env = ClusterEnvelope.parse(nodes)
    cat = load_catalog()
    if sql_file:
        q = query_from_sql(query or sql_file.stem, sql_file.read_text(), sf_eff)
    elif explain_file:
        q = query_from_explain(query or explain_file.stem, explain_file.read_text(), category)
    elif query:
        q = query_from_catalog(query, sf_eff, cat)
    else:
        raise typer.BadParameter("informe --query, --sql ou --explain")
    rec = recommend(q, sf_eff, env, objective=objective, top_k=top, data_gib=gib, catalog=cat)
    cols = ["profile", "workers", "total_cores", "pred_s", "core_hours", "cost_usd", "pareto", "measured_cell", "tie_with_best"]
    if as_json:
        out = {"query": q.name, "sf": sf_eff, "top": rec.top[cols].to_dict("records"),
               "warnings": rec.warnings, "regret_if_measured": regret_if_measured(rec)}
        typer.echo(json.dumps(out, indent=2, default=float))
        return
    typer.echo(f"consulta {q.name} ({q.category}, {q.source}) | SF {sf_eff:.0f} | objetivo {objective}")
    typer.echo(rec.top[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if rec.measured is not None:
        m = rec.measured.iloc[0]
        r = regret_if_measured(rec)
        extra = f"; regret da recomendação: {100*r:.1f}%" if r is not None else "; a recomendação não foi medida nesta escala"
        typer.echo(f"melhor célula medida: {m.profile} W{int(m.workers)} ({m.measured_s:.1f} s){extra}")
    for w in rec.warnings:
        typer.echo(f"aviso: {w}")


@app.command("catalog")
def cmd_catalog():
    cat = load_catalog()
    p = cat["plans"]
    typer.echo(f"{len(cat['queries'])} consultas, SFs {sorted(p.scale_factor.unique())}")
    typer.echo(p.groupby("query").first()[["query_category", "num_joins", "num_exchanges", "plan_lines"]].to_string())


def main():
    app()


if __name__ == "__main__":
    main()
