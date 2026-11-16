"""Ponto de entrada para o Streamlit Community Cloud.

Executa a aplicação em recommendation/web.py com a raiz do repositório no
caminho de importação.
"""
import runpy

runpy.run_module("recommendation.web", run_name="__main__")
