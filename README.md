# DS Flow

A Python library for data science workflows.

## Installation

```bash
pip install dsflow-py
```

## Usage

```python
# Import the main package
import ds_flow

# Import specific submodules
from ds_flow.torch_flow import my_function
from ds_flow.pandas_flow import my_other_function
from ds_flow.sk_flow import yet_another_function
```

## Features

- PyTorch workflows and utilities via `ds_flow.torch_flow`
- Pandas utilities via `ds_flow.pandas_flow`
- Scikit-learn tools via `ds_flow.sk_flow`

## Development

This project uses Poetry for dependency management.

```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run tests
poetry run pytest
```

## License

MIT