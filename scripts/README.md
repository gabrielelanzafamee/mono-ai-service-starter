# Docker Build and Push Scripts for Azure Container Registry

This directory contains scripts to build and push Docker images to Azure Container Registry (ACR).

## 📁 Scripts Overview

### 1. `build_image.sh` - Build Docker Image
Builds a Docker image and tags it for ACR deployment.

### 2. `push_image_to_acr.sh` - Push Image to ACR
Authenticates with ACR and pushes the built image.

### 3. `build_and_push.sh` - Combined Build and Push
Convenient script that builds and pushes in one command.

## 🚀 Quick Start

### Prerequisites

1. **Docker** installed and running
2. **Azure CLI** installed and authenticated
3. **Terraform** deployed ACR (see `infrastructure/README-ACR.md`)

### Basic Usage

```bash
# Build and push in one command (recommended)
./scripts/build_and_push.sh -a your-acr-name

# Or do it step by step
./scripts/build_image.sh -a your-acr-name
./scripts/push_image_to_acr.sh -a your-acr-name
```

## 📋 Detailed Usage

### Build Image Only

```bash
# Basic build
./scripts/build_image.sh -a myacr12345

# Custom image name and tag
./scripts/build_image.sh -n myapp -t v1.0.0 -a myacr12345

# Custom build context
./scripts/build_image.sh -a myacr12345 -c ./app
```

### Push Image Only

```bash
# Basic push (uses Azure CLI authentication)
./scripts/push_image_to_acr.sh -a myacr12345

# Use Docker login method
./scripts/push_image_to_acr.sh -a myacr12345 -m docker

# Custom image name and tag
./scripts/push_image_to_acr.sh -n myapp -t v1.0.0 -a myacr12345
```

### Combined Build and Push

```bash
# Basic build and push
./scripts/build_and_push.sh -a myacr12345

# Custom configuration
./scripts/build_and_push.sh -n myapp -t v1.0.0 -a myacr12345 -m docker
```

## 🔧 Script Options

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `-n, --name` | Docker image name | `mono-ai-service` |
| `-t, --tag` | Docker image tag | `latest` |
| `-a, --acr` | ACR name (required) | - |
| `-c, --context` | Build context path | `.` |
| `-m, --method` | Login method (`az` or `docker`) | `az` |
| `-h, --help` | Show help message | - |

### Authentication Methods

1. **Azure CLI (`az`)** - Recommended
   - Uses `az acr login`
   - Requires Azure CLI authentication
   - More secure and convenient

2. **Docker Login (`docker`)**
   - Uses `docker login` with credentials
   - Gets credentials from Azure CLI
   - Useful for CI/CD pipelines

## 📝 Examples

### Development Workflow

```bash
# 1. Deploy ACR with Terraform
cd infrastructure
terraform apply

# 2. Build and push your image
cd ..
./scripts/build_and_push.sh -a myacr12345

# 3. Pull and run locally for testing
docker pull myacr12345.azurecr.io/mono-ai-service:latest
docker run -p 8000:8000 myacr12345.azurecr.io/mono-ai-service:latest
```

### Version Management

```bash
# Build with version tag
./scripts/build_and_push.sh -n myapp -t v1.2.3 -a myacr12345

# Build with latest tag
./scripts/build_and_push.sh -n myapp -t latest -a myacr12345

# Build with environment tag
./scripts/build_and_push.sh -n myapp -t production -a myacr12345
```

### CI/CD Integration

```bash
# For GitHub Actions or Azure DevOps
./scripts/build_and_push.sh -a $ACR_NAME -n $IMAGE_NAME -t $TAG -m docker
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Docker Not Running
```
[ERROR] Docker is not running. Please start Docker and try again.
```
**Solution**: Start Docker Desktop or Docker daemon.

#### 2. Azure CLI Not Authenticated
```
[ERROR] Not logged in to Azure. Please run 'az login' first.
```
**Solution**: Run `az login` and follow the authentication process.

#### 3. ACR Name Not Found
```
[ERROR] Azure Container Registry name is required.
```
**Solution**: Provide the ACR name using `-a` option.

#### 4. Image Not Found Locally
```
[WARNING] Image not found locally.
```
**Solution**: Build the image first or use the last built image.

#### 5. Authentication Failed
```
[ERROR] Failed to authenticate with ACR.
```
**Solution**: 
- Check Azure permissions
- Try different authentication method
- Verify ACR exists and is accessible

### Debug Mode

Add `set -x` at the beginning of any script to enable debug output:

```bash
# Edit the script to add debug mode
sed -i '2i set -x' scripts/build_image.sh
```

## 📊 Script Features

### Build Script (`build_image.sh`)
- ✅ Validates Docker is running
- ✅ Checks for Dockerfile existence
- ✅ Supports custom build context
- ✅ Saves build information for later use
- ✅ Colored output for better UX
- ✅ Comprehensive error handling

### Push Script (`push_image_to_acr.sh`)
- ✅ Two authentication methods (Azure CLI, Docker login)
- ✅ Automatic credential retrieval
- ✅ Image existence validation
- ✅ Push history tracking
- ✅ Fallback to last built image
- ✅ Secure password handling

### Combined Script (`build_and_push.sh`)
- ✅ Orchestrates build and push process
- ✅ Consistent parameter handling
- ✅ Step-by-step progress reporting
- ✅ Final image verification

## 🔐 Security Best Practices

1. **Use Azure CLI authentication** when possible
2. **Don't commit credentials** to version control
3. **Use environment variables** for sensitive data in CI/CD
4. **Regularly rotate ACR credentials**
5. **Enable network rules** for production ACRs

## 📈 Performance Tips

1. **Use `.dockerignore`** to exclude unnecessary files
2. **Optimize Dockerfile** with proper layer caching
3. **Use multi-stage builds** for smaller images
4. **Consider ACR Tasks** for automated builds
5. **Use appropriate ACR SKU** for your needs

## 🎯 Next Steps

After successfully building and pushing your image:

1. **Deploy to Azure App Service** using the ACR image
2. **Set up CI/CD pipelines** for automated deployments
3. **Configure monitoring** for your containerized application
4. **Implement blue-green deployments** for zero-downtime updates
5. **Set up ACR Tasks** for automated builds on code changes 