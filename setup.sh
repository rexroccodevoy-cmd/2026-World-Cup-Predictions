#!/bin/bash
# 2026 World Cup Predictor — one-time setup
set -e

echo "Installing dependencies..."
pip install streamlit pandas numpy scikit-learn scipy plotly statsmodels xgboost --upgrade

echo ""
echo "✅ Done. Launch the dashboard with:"
echo ""
echo "   streamlit run app.py"
echo ""
