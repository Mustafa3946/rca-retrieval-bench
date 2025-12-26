# Contributing to Quality–Latency Benchmarking Framework

Thank you for your interest in contributing to this project!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/LAR_RAG.git
cd LAR_RAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Optional: for testing and linting
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_temporal_scoring.py

# Run with coverage
pytest --cov=src tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Keep functions focused and well-documented

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add some feature'`)
7. Push to the branch (`git push origin feature/your-feature`)
8. Open a Pull Request

## Areas for Contribution

- **Documentation**: Improve or expand documentation
- **Examples**: Add more usage examples
- **Tests**: Increase test coverage
- **Features**: 
  - New Stage 1 retrievers
  - Alternative scoring functions
  - Performance optimizations
  - Additional datasets/benchmarks

## Questions?

Feel free to open an issue for questions, bug reports, or feature requests.
