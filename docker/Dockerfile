# Use PyTorch base image which already has PyTorch installed
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Install flash-attn first (since it needs PyTorch to be available)
RUN pip install flash-attn --no-build-isolation
RUN pip install bitsandbytes accelerate

# Install the rest of the dependencies with Poetry
RUN poetry install --no-root --no-interaction --no-ansi

RUN PIP_NO_BUILD_ISOLATION=1 poetry run python -m pip install flash-attn
RUN PIP_NO_BUILD_ISOLATION=1 poetry run python -m pip install bitsandbytes accelerate

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Entrypoint is the python, the user will choose what to run if the server or train/main.py
ENTRYPOINT ["poetry", "run"]