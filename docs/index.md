# opencode applied

Welcome to the comprehensive documentation portal for the Stock Clustering Analysis project. This site provides complete guidance from project overview through implementation to deployment.

## 🚀 Quick Start

### Try the Docker Demo
```bash
# Pull the automated build from GitHub Container Registry
docker pull ghcr.io/najnesnaj/clustering:latest

# Run the container with port mapping
docker run -p 8501:8501 ghcr.io/najnesnaj/clustering:latest

# Access immediately at: http://localhost:8501
```

### Build Locally
```bash
# Clone and build the container
git clone https://github.com/najmus-saqib/clustering.git
cd clustering
docker build -t clustering-demo .
docker run -p 8501:8501 clustering-demo
```

## 📚 Documentation Structure

This documentation is organized into three main sections:

### 📖 [Project Overview](overview/project-intro.md)
- **Introduction** - Complete project overview and capabilities
- **Docker Demo** - Comprehensive Docker deployment guide with automated CI/CD

### 🛠️ [Planning & Development](planning/initial-plan.md)
- **Initial Plan** - Original project requirements and technical architecture
- **Docker Planning** - Specific Docker container planning and design
- **Setup Documentation** - Implementation details and technical decisions
- **Implementation Complete** - Project completion verification and results

### 🚀 [Deployment & Usage](guides/docker-deployment.md)
- **Docker Deployment** - Step-by-step deployment instructions
- **Database Setup** - PostgreSQL configuration and data management
- **Notebook Compatibility** - Differences between research and production
- **Demo Results** - Example outputs and next steps

## 🎯 Project Capabilities

### Data Processing
- **15 diversified stocks** from various sectors
- **5 years of historical data** with realistic patterns
- **6 technical features** per stock (returns, volatility, RSI, etc.)
- **SQLite embedded database** for instant access

### Machine Learning
- **KMeans clustering** with automatic optimal cluster detection
- **Silhouette analysis** for best cluster count
- **Feature scaling** and normalization
- **Descriptive labeling** for each cluster

### Interactive Dashboard
- **Streamlit interface** with Plotly visualizations
- **Real-time analysis** and exploration
- **Professional charts** and time series analysis
- **Export capabilities** for results

### Deployment Features
- **Automated CI/CD** via GitHub Actions
- **Multi-platform support** (amd64/arm64)
- **Container Registry** integration (GitHub Container Registry)
- **Zero configuration** deployment

## 🔧 Technical Stack

- **Backend**: Python 3.11 with Scikit-learn
- **Frontend**: Streamlit with Plotly visualizations
- **Database**: SQLite (embedded, zero config)
- **Containerization**: Docker with GitHub Actions
- **Documentation**: MkDocs with Material theme
- **Deployment**: GitHub Pages and GitHub Container Registry

---

!!! tip "Getting Started"
    New users should start with the [Project Overview](overview/project-intro.md) to understand the system capabilities, then proceed to [Deployment & Usage](guides/docker-deployment.md) for hands-on implementation.

!!! info "For Developers"
    Developers interested in the technical implementation should review the [Planning & Development](planning/initial-plan.md) section to understand the architecture and decision-making process.