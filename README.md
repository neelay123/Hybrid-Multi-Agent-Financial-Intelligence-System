# Hybrid Multi-Agent Financial Intelligence System

A comprehensive AI system integrating fine-tuned language models with graph neural networks for enhanced financial analysis, built with SEC EDGAR data processing, multi-modal embeddings, and intelligent agent orchestration.

## Overview

This repository contains three main components:
1. **Training Pipeline** (`financial_gnn_training_pipeline.py`) - Data processing, knowledge graph construction, and GNN model training
2. **Inference System** (`unified_financial_ai_system.py`) - Multi-agent financial intelligence system with LangGraph orchestration
3. **Model Evaluation** (`phi4_evaluation_script.py`) - Comparative evaluation of base vs fine-tuned Phi-4 models

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID FINANCIAL AI SYSTEM                  │
├─────────────────────────────────────────────────────────────────┤
│  Fine-tuned Phi-4 LLM  │  Knowledge Graph  │  Multi-Agent Flow │
│  • LoRA Adaptation      │  • SEC EDGAR Data │  • LangGraph      │
│  • FinQA Training       │  • GNN Embeddings │  • Query Routing  │
│  • 26% Accuracy         │  • 420K Relations │  • Entity Resolve │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### System Requirements
- Python 3.9+
- CUDA-compatible GPU (recommended: 16GB+ VRAM)
- 32GB+ RAM
- 50GB+ free disk space

### Hardware Recommendations
- **Training Phase**: RTX 4090 / A6000 or equivalent
- **Inference Phase**: RTX 4080 / V100 or equivalent
- **Memory**: Training uses 90% GPU memory, inference uses ~60%

## Installation

### 1. Clone Repository
```bash
git clone <your-repository-url>
cd hybrid-financial-ai-system
```

### 2. Create Virtual Environment
```bash
python -m venv financial_ai_env
source financial_ai_env/bin/activate  # On Windows: financial_ai_env\Scripts\activate
```

### 3. Install Dependencies
```bash
# Core ML and AI libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Transformers and fine-tuning
pip install transformers>=4.35.0
pip install peft>=0.6.0
pip install trl>=0.7.0
pip install datasets>=2.14.0

# Graph processing
pip install networkx>=3.0
pip install qdrant-client>=1.6.0
pip install faiss-cpu  # or faiss-gpu for GPU acceleration

# Multi-agent orchestration
pip install langgraph>=0.0.40
pip install langchain>=0.1.0
pip install langchain-core>=0.1.0

# Data processing and utilities
pip install pandas numpy matplotlib
pip install requests zipfile36
pip install cachetools
pip install scikit-learn

# Optional: Jupyter for interactive development
pip install jupyter notebook ipywidgets
```

### 4. Set Environment Variables
Create a `.env` file in the project root:
```env
# Optional: API keys for enhanced data (not required for basic functionality)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
SEC_API_KEY=your_sec_api_key_here

# GPU memory management
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
TOKENIZERS_PARALLELISM=true
```

## Usage

### Phase 1: Training Pipeline (Data Processing & Model Training)

**Purpose**: Download SEC EDGAR data, build knowledge graph, train GNN models, generate embeddings

```bash
python financial_gnn_training_pipeline.py
```

**What this does**:
1. Downloads ~18,000 SEC EDGAR company filings (may take 2-4 hours)
2. Processes and enriches company data with financial metrics
3. Builds knowledge graph with 17,552 nodes and 420,796 relationships
4. Trains 4 GNN models: GraphSAGE, Attention GNN, Temporal GNN, E5 embeddings
5. Stores embeddings in Qdrant vector database
6. Saves trained models to `trained_models_YYYYMMDD_HHMMSS/` directory

**Expected Output**:
```
📊 System Status: {
    'available_embedding_models': ['graphsage', 'attention_gnn', 'temporal_gnn', 'e5'],
    'total_entities_loaded': 16333,
    'gnn_model_loaded': True,
    'graphrag_engine_ready': True
}
```

**Training Time**: 3-6 hours (depending on hardware)
**Disk Usage**: ~5GB for processed data + ~2GB for models

### Phase 2: Interactive Financial AI System

**Purpose**: Run the complete multi-agent financial intelligence system

```bash
python unified_financial_ai_system.py
```

**System Capabilities**:
- **Query Types Supported**:
  - Global: "What is Apple's P/E ratio?"
  - Local: "Find companies similar to Tesla"
  - Numerical: "Calculate profit margin for $500M revenue, $100M profit"

**Example Interaction**:
```
🚀 Initializing Financial AI System...
✅ System is ready. Enter your financial query or type 'exit' to quit.

> Find companies similar to Apple Inc
Processing your query...
  -> Step completed: query_analysis
  -> Step completed: gnn_inference
  -> Step completed: generate_response

================ FINAL RESPONSE ================
Based on semantic analysis, here are the top competitors for Apple Inc.:

1. Microsoft Corporation
2. Samsung Electronics Co., Ltd.
3. Alphabet Inc.
4. Meta Platforms, Inc.
5. Amazon.com, Inc.
6. NVIDIA Corporation
7. Intel Corporation
8. Advanced Micro Devices, Inc.
9. Sony Group Corporation
10. Tesla, Inc.
================================================
```

### Phase 3: Model Evaluation (Optional)

**Purpose**: Compare base Phi-4 vs fine-tuned model performance

**Prerequisites**: 
1. Download FinQA test data to `finqa_data/test.json`
2. Have fine-tuned model in `./phi4-finqa-final/` directory

```bash
python phi4_evaluation_script.py
```

**Expected Output**:
```
FINAL SUMMARY
================
Total samples evaluated: 100

--- EXACT MATCH RESULTS ---
Base model exact correct: 4/100 (4.0%)
Fine-tuned model exact correct: 26/100 (26.0%)
🎉 EXACT MATCH WINNER: Fine-tuned model! (26 vs 4)

--- NUMERICAL ACCURACY RESULTS (±1%) ---
Base model numerical correct: 14/100 (14.0%)
Fine-tuned model numerical correct: 66/100 (66.0%)
🎉 NUMERICAL ACCURACY WINNER: Fine-tuned model! (66 vs 14)
```

## Directory Structure

```
hybrid-financial-ai-system/
├── README.md
├── .env
├── financial_gnn_training_pipeline.py    # Main training script
├── unified_financial_ai_system.py        # Interactive AI system
├── phi4_evaluation_script.py             # Model evaluation
├── finqa_data/                           # FinQA dataset (download separately)
│   └── test.json
├── edgar_data/                           # SEC EDGAR filings (auto-downloaded)
├── trained_models_YYYYMMDD_HHMMSS/       # Trained model outputs
│   ├── graphsage.pth
│   ├── attention_gnn.pth
│   ├── temporal_gnn.pth
│   ├── embeddings.pkl
│   ├── knowledge_graph.pkl
│   └── training_metadata.json
├── financial_embeddings_db/              # Qdrant vector database
├── phi4-finqa-final/                     # Fine-tuned Phi-4 model (if available)
├── enriched_companies.jsonl             # Processed company data
├── sec_api_cache.json                    # SEC API response cache
└── offload/                              # Model offloading directory
```

## Configuration Options

### GPU Memory Management
```python
# In SystemConfig class
self.gpu_memory_fraction = 0.9  # Adjust based on your GPU
self.max_gpu_memory = "32GiB"   # Set your GPU memory limit
```

### Training Parameters
```python
# In SystemConfig class
self.training_epochs = 50       # Increase for better convergence
self.learning_rate = 0.005     # Adjust learning rate
self.weight_decay = 5e-4       # Regularization strength
```

### Data Processing Limits
```python
# In training pipeline
edgar_downloader.parse_company_facts(companyfacts_dir, limit=5000)  # Limit for testing
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce GPU memory fraction
# In SystemConfig: self.gpu_memory_fraction = 0.7

# Or use CPU for some components
device = "cpu"  # Force CPU usage
```

**2. SEC API Rate Limiting**
```
Solution: The system implements automatic retry with exponential backoff
Wait time: 1-32 seconds between retries
Max retries: 5 per request
```

**3. Missing Dependencies**
```bash
# For torch-geometric issues:
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
pip install torch-geometric pyg-lib torch-scatter torch-sparse torch-cluster --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

**4. Qdrant Connection Issues**
```python
# Falls back to in-memory mode automatically
# Check: ./financial_embeddings_db/ directory permissions
```

**5. Model Loading Errors**
```bash
# Ensure models are in correct directories:
ls ./trained_models_*/  # Should contain .pth files
ls ./phi4-finqa-final/  # Should contain adapter files
```

### Performance Optimization

**Memory Optimization**:
- Use `torch.cuda.empty_cache()` between operations
- Implement gradient checkpointing for large models
- Use streaming data processing for large datasets

**Speed Optimization**:
- Enable flash attention 2.0
- Use bfloat16 precision
- Implement model parallelism for multiple GPUs

## System Monitoring

### Check System Status
```python
# In interactive mode, the system reports:
📊 System Status: {
    "available_embedding_models": ["graphsage", "attention_gnn", "temporal_gnn", "e5"],
    "total_entities_loaded": 16333,
    "gnn_model_loaded": True,
    "graphrag_engine_ready": True
}
```

### Performance Metrics
- **Query Response Time**: <5 seconds average
- **Memory Usage**: 90% GPU utilization during training, ~60% during inference
- **Entity Resolution**: 100% success rate
- **Query Classification**: 100% accuracy

## API Reference

### Key Functions

```python
# Training Pipeline
pipeline = FinancialGNNTrainingPipeline(SystemConfig())
results = pipeline.run_full_training_pipeline()

# Inference System  
config = SystemConfig()
system = LangGraphFinancialSystem(config)

# Query the system
for event in system.stream_query("Find companies similar to Apple"):
    print(event)
```

### Core Classes

- `SystemConfig`: System configuration and hyperparameters
- `FinancialGNNTrainingPipeline`: Complete training orchestration
- `LangGraphFinancialSystem`: Multi-agent inference system
- `UnifiedKnowledgeGraphAgent`: Knowledge graph operations
- `FineTunedPhi4Agent`: Fine-tuned language model interface

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@article{hybrid_financial_ai_2024,
  title={Hybrid Multi-Agent Financial Intelligence System: Integrating Fine-Tuned Language Models with Graph Neural Networks for Enhanced Financial Analysis},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review system logs in console output
3. Ensure all dependencies are correctly installed
4. Verify GPU/CUDA compatibility

**Note**: This system is designed for research and educational purposes. For production financial applications, additional security, compliance, and validation measures should be implemented.
