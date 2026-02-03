# 📓 Notebook Compatibility Note

## 🚧 **Important Information**

The Jupyter notebooks in the `/notebooks/` directory were created for the **original research project** and may require:

1. **PostgreSQL database** connection (not used in Docker demo)
2. **Yahoo Finance API** access (replaced with mock data in Docker)
3. **Full clustering modules** with tslearn (simplified in Docker)

## ✅ **Docker Demo Solution**

The Docker container uses **simplified, self-contained approach**:
- **Mock stock data** instead of yfinance API
- **SQLite database** instead of PostgreSQL
- **Simplified clustering** without tslearn
- **Streamlit dashboard** for production use

## 📓 **For Notebook Users**

If you want to run the exploratory notebooks:

```bash
# Install all dependencies
pip install -r requirements.txt

# Set up PostgreSQL database (required for original notebooks)
# Configure connection in config/database.py

# Notebooks expect:
# - PostgreSQL database with stock symbols
# - Yahoo Finance API access
# - Full clustering pipeline
```

## 🎯 **Recommendation**

**Use the Streamlit dashboard** (`app.py`) for the best experience:
- ✅ Works immediately (no setup required)
- ✅ Interactive and professional
- ✅ Production-ready visualizations
- ✅ Self-contained (no external dependencies)

The notebooks are provided for **reference and exploration** of the original research methodology.