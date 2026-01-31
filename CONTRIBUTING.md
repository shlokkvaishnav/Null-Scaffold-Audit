# Contributing to SD-MoSE

Thank you for your interest in contributing to SD-MoSE! We welcome contributions from the scientific and open-source community.

## 🛠Development Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/shlokkvaishnav/climate-equation-discovery.git
    cd climate-equation-discovery
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

3.  **Install pre-commit hooks**
    ```bash
    pre-commit install
    ```

## 🧪 Running Tests

Ensure all tests pass before submitting a pull request:

```bash
pytest tests/
```

## 📝 Code Style

We follow PEP 8 and use `black` for formatting.
- **Linting**: `flake8`
- **Formatting**: `black`
- **Type Checking**: `mypy`

## 🤝 Pull Request Process

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes.
4.  Push to the branch.
5.  Open a Pull Request.

## 🔬 Research Contributions

If you are adding new physical constraints or datasets:
1.  Document the scientific basis in the PR description.
2.  Ensure new features are unit tested.
3.  Update `CITATION.cff` if you are a co-author.
