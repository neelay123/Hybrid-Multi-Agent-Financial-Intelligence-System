# Hybrid Multi-Agent Financial Intelligence System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.
[![PyTorch](https://img.shields.io/lds.A cutting-edge financial AI system integrating fine-tuned language models with graph neural networks for comprehensive financial analysis and intelligent decision-making.**

## 🚀 Overview

This project presents a hybrid multi-agent AI system that revolutionizes financial analysis by combining three core components:

- **🧠 Domain-Specific Fine-Tuned Phi-4 Model** - Enhanced with LoRA for financial reasoning
- **🕸️ Large-Scale Financial Knowledge Graph** - Built from 18,000+ SEC EDGAR filings
- **🤖 Multi-Agent Orchestration** - Intelligent query routing using LangGraph

The system achieves **550% improvement** in financial question-answering accuracy and processes **68,989 embeddings** across four model architectures for comprehensive financial intelligence.

## ✨ Key Features

### 🎯 **Advanced Financial Reasoning**
- Fine-tuned Microsoft Phi-4 (14.7B parameters) on FinQA dataset
- Two-stage numerical reasoning for precise financial calculations
- Multi-step reasoning with step-by-step explanations

### 🌐 **Comprehensive Knowledge Graph**
- **17,552 nodes** and **420,796 relationships**
- Multi-architecture GNN training (GraphSAGE, GATv2, Temporal)
- Real-time SEC EDGAR and Yahoo Finance integration

### 🔍 **Intelligent Query Processing**
- **100% query routing accuracy** across 3 query types
- Waterfall entity resolution (exact → semantic → fuzzy)
- **Sub-5-second response times** with caching optimization

### 🏗️ **Production-Ready Architecture**
- Memory-efficient streaming data processing
- Scalable vector storage with Qdrant
- Robust error handling and graceful degradation

## 🏆 Performance Highlights

| Metric | Base Model | Fine-Tuned Model | Improvement |
|--------|------------|------------------|-------------|
| **Exact Match Accuracy** | 4.0% | 26.0% | **+550%** |
| **Numerical Accuracy (±1%)** | 14.0% | 66.0% | **+371%** |
| **Entity Resolution** | - | 100% | **Perfect** |
| **Query Routing** | - | 100% | **Perfect** |

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended: 32GB+ VRAM)
- 64GB+ RAM for full dataset processing

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/neelaychoudhury/hybrid-financial-ai.git
cd hybrid-financial-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional graph libraries
pip install torch-geometric torch-sparse torch-scatter
```

### Environment Variables

Create a `.env` file in the root directory:

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_key_here
SEC_API_KEY=your_key_here  # Optional

# Model Paths
PHI4_MODEL_PATH=microsoft/phi-4
FINE_TUNED_MODEL_PATH=./phi4-finqa-final

# System Configuration
GPU_MEMORY_FRACTION=0.9
MAX_WORKERS=5
CACHE_TTL=3600
```

## 🚀 Quick Start

### 1. Data Preparation

```python
from src.data_processing import EDGARDataDownloader, KaggleXBRLDataEnricher

# Download SEC EDGAR data
downloader = EDGARDataDownloader()
extracted_dir = downloader.download_bulk_data("companyfacts")

# Enrich with market data
enricher = KaggleXBRLDataEnricher()
enricher.process_and_save_data("./enriched_companies.jsonl")
```

### 2. Train Models

```python
from src.training import FinancialGNNTrainingPipeline
from src.config import SystemConfig

# Initialize training pipeline
config = SystemConfig()
pipeline = FinancialGNNTrainingPipeline(config)

# Run complete training
results = pipeline.run_full_training_pipeline()
```

### 3. Launch the System

```python
from src.system import LangGraphFinancialSystem
from src.config import SystemConfig

# Initialize the complete system
config = SystemConfig()
system = LangGraphFinancialSystem(config)

# Interactive query processing
while True:
    query = input("\n> ")
    if query.lower() in ['exit', 'quit']:
        break
    
    # Process query through multi-agent workflow
    for event in system.stream_query(query):
        if "__end__" in event:
            final_response = event["messages"][-1].content
            print(f"\n📊 {final_response}")
            break
```

## 🏗️ System Architecture
<img width="3727" height="3840" alt="Untitled diagram _ Mermaid Chart-2025-08-21-005335" src="https://github.com/user-attachments/assets/7400e8a0-f0fa-4765-972f-1f2b719f0353" />


<img width="2105" height="3840" alt="Untitled diagram _ Mermaid Chart-2025-08-21-012331" src="https://github.com/user-attachments/assets/8b302a77-f1d6-4b20-b2c8-51e094e736f1" />


<img width="3840" height="2352" alt="Untitled diagram _ Mermaid Chart-2025-08-21-014841" src="https://github.com/user-attachments/assets/df87e079-8fea-4d2f-addd-e90aa910db92" />

## 📊 Dataset Information

### SEC EDGAR Processing
- **Raw Files**: 18,877 company fact files
- **Successfully Parsed**: 16,333 companies (86.5% success rate)
- **Final Graph Nodes**: 17,552 entities
- **Relationships**: 420,796 edges

### Knowledge Graph Statistics
| Component | Count | Description |
|-----------|-------|-------------|
| **Company Nodes** | 16,333 | Public companies with SEC filings |
| **Industry Nodes** | 372 | SIC-based industry classifications |
| **Metric Nodes** | 847 | Financial metrics and ratios |
| **Peer Relationships** | 244,995 | Industry-based connections |
| **Market Cap Similarities** | 81,573 | Size-based relationships |

## 🎯 Usage Examples

### Financial Entity Analysis
```python
# Find competitors for a specific company
query = "Find companies similar to Apple Inc"
response = system.process_query(query)
# Returns: Top 10 semantically and structurally similar companies
```

### Numerical Financial Calculations
```python
# Complex financial calculations
query = "If a company has revenue of $1.2B and profit margin of 15%, what is the net income?"
response = system.process_query(query)
# Returns: Step-by-step calculation with final answer
```

### Market Data Retrieval
```python
# Real-time financial metrics
query = "What is Tesla's current P/E ratio?"
response = system.process_query(query)
# Returns: Live data from Yahoo Finance with source attribution
```

## 🔧 Advanced Configuration

### Model Fine-Tuning Parameters

```python
# LoRA Configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"]
}

# Training Configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "warmup_ratio": 0.03
}
```

### GNN Architecture Options

```python
# Available GNN Models
GNN_MODELS = {
    "graphsage": {
        "hidden_dim": 512,
        "output_dim": 1024,
        "num_layers": 2
    },
    "attention_gnn": {
        "hidden_dim": 128,
        "num_heads": 8,
        "num_layers": 3
    },
    "temporal_gnn": {
        "periods": 12,
        "hidden_dim": 64,
        "attention_heads": 4
    }
}
```

## 📈 Benchmarks & Performance

### Query Processing Performance
- **Average Response Time**: < 5 seconds
- **Entity Resolution Accuracy**: 100%
- **Query Routing Accuracy**: 100%
- **System Uptime**: > 95%

### Memory & Computational Requirements
- **GPU Memory Usage**: 90% of available (optimized)
- **RAM Usage**: 24GB+ for full dataset processing
- **Storage**: 50GB+ for complete embeddings

### Industry-Specific Performance
| Sector | Companies Tested | Relevance Score |
|--------|-----------------|-----------------|
| **Banking/Finance** | 3 | 95% |
| **Technology** | 3 | 90% |
| **Automotive** | 2 | 85% |
| **Pharmaceutical** | 1 | 90% |
| **Retail** | 1 | 95% |

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Performance benchmarks
python -m pytest tests/benchmarks/ -v

# End-to-end system tests
python tests/test_system_e2e.py
```

## 📝 API Reference

### Core Classes

#### `LangGraphFinancialSystem`
Main system orchestrator handling multi-agent workflows.

```python
system = LangGraphFinancialSystem(config)
results = system.stream_query("Find Apple competitors")
```

#### `UnifiedKnowledgeGraphAgent`
Manages graph operations and entity resolution.

```python
kg_agent = UnifiedKnowledgeGraphAgent(config, embedder)
entities = kg_agent.resolve_entity("Apple Inc")
similar = kg_agent.find_similar_entities("Apple Inc", top_k=10)
```

#### `FineTunedPhi4Agent`
Handles language model operations and reasoning.

```python
phi4_agent = FineTunedPhi4Agent(config)
response = phi4_agent.generate(prompt, max_new_tokens=200)
analysis = phi4_agent.analyze_query(query)
```


## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/hybrid-financial-ai.git

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest
```

### Code Style

We follow [Black](https://black.readthedocs.io/) for code formatting and [flake8](https://flake8.pycqa.org/) for linting.

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

## 📚 Documentation

- [**Academic Paper**](docs/paper.md) - Complete research documentation
- [**API Documentation**](docs/api.md) - Detailed API reference
- [**Architecture Guide**](docs/architecture.md) - System design deep-dive
- [**Deployment Guide**](docs/deployment.md) - Production deployment instructions

## ⚖️ Legal & Compliance

### Data Usage
- SEC EDGAR data is publicly available and used in compliance with SEC guidelines
- All financial data is used for research and educational purposes
- No proprietary or confidential information is included

### Model Licensing
- Base models (Phi-4, E5-Large) used under their respective licenses
- Fine-tuned models and system code released under MIT License
- Commercial use permitted with proper attribution

### Disclaimer
This system is for research and educational purposes. Not intended as financial advice. Users should consult qualified financial professionals for investment decisions.

## 🙏 Acknowledgments

- **Microsoft Research** for the Phi-4 base model
- **Hugging Face** for transformers and training frameworks
- **PyTorch Geometric** team for graph neural network implementations
- **SEC EDGAR** for providing comprehensive financial data
- **Queen Mary University of London** and supervisor **Shalom Lappin** for academic support

## 📞 Contact & Support

- **Author**: Neelay Choudhury
- **LinkedIn**: [linkedin.com/in/neelaychoudhury]([https://linkedin.com/in/neelaychoudhury](https://www.linkedin.com/in/neelay-choudhury-768537152/))
- **Project Issues**: [GitHub Issues](https://github.com/neelaychoudhury/hybrid-financial-ai/issues)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

***

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

[![Star History Chart](https://api.star-history.com/svg?repos=neelaychoudhury/hybriBuilt with ❤️ for the future of financial AI*

</div>
