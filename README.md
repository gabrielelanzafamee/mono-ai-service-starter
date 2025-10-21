# Mono AI Service Starter

A comprehensive starter template for building AI-powered services with fine-tuned language models, FastAPI backend, and cloud infrastructure deployment.

## 🚀 Features

- **LLM Fine-tuning**: Parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
- **FastAPI Backend**: Production-ready API service with modern Python stack
- **Cloud Infrastructure**: Azure deployment with Terraform Infrastructure as Code
- **Containerization**: Docker support for consistent deployment
- **CI/CD Pipeline**: GitHub Actions workflow for automated deployment
- **Data Management**: Structured data handling with Co-op product datasets

## 📁 Project Structure

```
mono-ai-service-starter/
├── train/                      # AI model training components
│   ├── models.py              # LLM wrapper with LoRA support
│   ├── trainer.py             # Training pipeline using TRL
│   ├── preprocess.py          # Dataset preprocessing utilities
│   ├── evaluate.py            # Model evaluation framework
│   └── main.py                # Training script entry point
├── data/                       # Training and validation datasets
├── infrastructure/             # Terraform IaC for Azure deployment
│   ├── main.tf                # Main infrastructure configuration
│   ├── outputs.tf             # Infrastructure outputs
│   └── terraform.tf           # Provider configuration
├── notebooks/                  # Jupyter notebooks for experimentation
├── .github/workflows/          # CI/CD pipeline configuration
├── main.py                     # FastAPI application entry point
├── Dockerfile                  # Container configuration
├── pyproject.toml             # Python dependencies and project metadata
└── README.md                   # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- Docker (for containerization)
- Azure CLI (for cloud deployment)
- Terraform (for infrastructure management)

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd mono-ai-service-starter
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Install bitsandbytes, accelerate and flash-attn**
   ```bash
   PIP_NO_BUILD_ISOLATION=1 poetry run python -m pip install flash-attn
   PIP_NO_BUILD_ISOLATION=1 poetry run python -m pip install bitsandbytes accelerate
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the FastAPI service:**
   ```bash
   poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 🤖 AI Model Training

### Quick Start

Train a model using the provided training pipeline:

```bash
cd train
poetry run python main.py
```

### Custom Training

```python
from train.trainer import Trainer

trainer = Trainer(
    model_id="google/gemma-3-1b-it",
    dataset_id="your-dataset-id",
    output_model_id="your-output-model-id"
)
trainer.train()
```

### Model Architecture

The project uses **Gemma 3 1B** as the base model with:
- **LoRA Configuration**: Rank 64, Alpha 16, 5% dropout
- **Target Modules**: All attention and MLP layers
- **Training Strategy**: Supervised Fine-Tuning (SFT) with TRL
- **Optimization**: Paged AdamW with gradient accumulation

### Training Configuration

- **Batch Size**: 1 per device with 2x gradient accumulation
- **Learning Rate**: 2e-4 with constant scheduler
- **Epochs**: 15 with early stopping
- **Mixed Precision**: BF16 on GPU, FP16 fallback
- **Monitoring**: Weights & Biases integration

## 🌐 API Service

### FastAPI Application

The main API service (`main.py`) provides:

- **Health Check**: `GET /` - Basic service status
- **Environment Configuration**: Automatic port detection
- **Production Ready**: ASGI server with Uvicorn

### Running the Service

```bash
# Development
poetry run uvicorn main:app --reload

# Production
poetry run uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🏗️ Infrastructure

### Azure Deployment

The project includes complete Azure infrastructure:

- **App Service**: Linux-based web app with Docker support
- **Container Registry**: Private Docker image storage
- **Cosmos DB**: NoSQL database with MongoDB API
- **Resource Group**: Organized resource management

### Terraform Deployment

1. **Initialize Terraform:**
   ```bash
   cd infrastructure
   terraform init
   ```

2. **Plan deployment:**
   ```bash
   terraform plan
   ```

3. **Apply infrastructure:**
   ```bash
   terraform apply
   ```

### Infrastructure Components

- **App Service Plan**: Basic B1 tier for cost optimization
- **Web App**: Linux container with automatic deployment
- **ACR**: Container registry with admin access
- **Cosmos DB**: MongoDB API with session consistency

## 🐳 Docker Support

### Building the Image

```bash
docker build -t mono-ai-service-starter .
```

### Running the Container

```bash
docker run -p 8000:8000 mono-ai-service-starter
```

### Multi-stage Build

The Dockerfile uses Poetry for dependency management and includes:
- Python 3.12 base image
- Poetry installation and caching
- Application code copying
- Port exposure (8000)

## 📊 Data Management

### Datasets

The project includes structured datasets for:

- **Product Catalog**: Co-op member products with pricing
- **Validation Cases**: Test scenarios for meal planning
- **Training Data**: Instruction-following examples

### Data Processing

```python
from train.preprocess import Dataset

dataset = Dataset("your-dataset-id")
train_data = dataset.train
test_data = dataset.test
```

## 🔧 Dependencies

### Core Dependencies

- **FastAPI**: Modern web framework
- **Transformers**: HuggingFace model library
- **PyTorch**: Deep learning framework
- **TRL**: Transformer Reinforcement Learning
- **PEFT**: Parameter-Efficient Fine-Tuning
- **SQLModel**: Database ORM
- **Uvicorn**: ASGI server

### Development Dependencies

- **Poetry**: Dependency management
- **Weights & Biases**: Experiment tracking
- **Datasets**: Data loading and processing

## 🚀 CI/CD Pipeline

### GitHub Actions

The project includes a comprehensive CI/CD pipeline:

- **Build & Test**: Code quality checks
- **Docker Build**: Container image creation
- **Azure Deployment**: Automated infrastructure deployment

### Deployment Strategy

1. **Development**: Local testing and validation
2. **Staging**: Container registry push
3. **Production**: Azure App Service deployment

## 📈 Monitoring & Evaluation

### Model Evaluation

The evaluation framework supports:

- **Baseline Comparison**: Original vs fine-tuned models
- **Reasoning Evaluation**: LLM-based assessment
- **Metrics Tracking**: Automated performance monitoring

### Service Monitoring

- **Health Checks**: Built-in endpoint monitoring
- **Logging**: Structured application logs
- **Metrics**: Performance and usage tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋 Support

For questions and support:

- **Author**: Gabriele Lanzafame
- **Issues**: GitHub Issues page

## 🔮 Future Enhancements

- [ ] Multi-model support
- [ ] Advanced evaluation metrics
- [ ] Kubernetes deployment
- [ ] Real-time inference optimization
- [ ] A/B testing framework

---

**Note**: This is a starter template. Customize the configuration, datasets, and infrastructure according to your specific use case and requirements.
