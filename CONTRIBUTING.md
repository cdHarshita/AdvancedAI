# Contributing to Advanced AI Systems Lab

Thank you for your interest in contributing! This project aims to be a **portfolio-quality enterprise AI platform**, so we maintain high standards for code quality and architecture.

## 🎯 Contribution Philosophy

We value:
1. **Production-quality code** over quick prototypes
2. **Security** over convenience
3. **Maintainability** over cleverness
4. **Best practices** over "it works"
5. **Documentation** over self-documenting code

## 📋 Before You Start

1. **Read the review guidelines**:
   - [Architectural Review Guide](docs/review-guidelines/ARCHITECTURAL_REVIEW.md)
   - [Security Review Guide](docs/review-guidelines/SECURITY_REVIEW.md)
   - [RAG Review Guide](docs/review-guidelines/RAG_REVIEW.md)

2. **Understand the automated review system**:
   - Every PR is reviewed by our AI Systems Architect bot
   - Address all critical issues before requesting human review
   - Security vulnerabilities block PRs

3. **Check existing issues**:
   - Look for existing issues or PRs
   - Comment if you want to work on something
   - Avoid duplicate work

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repo on GitHub first, then:
git clone https://github.com/YOUR-USERNAME/AdvancedAI.git
cd AdvancedAI
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys (NEVER commit this file!)
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Use meaningful branch names:
- `feature/langchain-rag-implementation`
- `fix/memory-leak-in-agent`
- `docs/update-rag-guide`
- `refactor/improve-error-handling`

## 📝 Making Changes

### Code Standards

#### Security (Critical)
- ✅ **DO**: Use environment variables for secrets
- ✅ **DO**: Validate all user inputs
- ✅ **DO**: Sanitize LLM outputs
- ❌ **DON'T**: Hardcode API keys
- ❌ **DON'T**: Commit `.env` files
- ❌ **DON'T**: Log sensitive data

#### Architecture (Important)
- ✅ **DO**: Follow SOLID principles
- ✅ **DO**: Separate concerns
- ✅ **DO**: Use dependency injection
- ✅ **DO**: Write testable code
- ❌ **DON'T**: Create god classes
- ❌ **DON'T**: Tight coupling

#### AI Best Practices
- ✅ **DO**: Set memory limits
- ✅ **DO**: Add agent guardrails (max_iterations)
- ✅ **DO**: Implement retry logic
- ✅ **DO**: Add monitoring callbacks
- ✅ **DO**: Optimize chunk sizes
- ❌ **DON'T**: Use unbounded buffers
- ❌ **DON'T**: Skip error handling
- ❌ **DON'T**: Ignore costs

#### Code Quality
- ✅ **DO**: Write self-documenting code
- ✅ **DO**: Add docstrings for classes and functions
- ✅ **DO**: Follow PEP 8
- ✅ **DO**: Keep functions small and focused
- ❌ **DON'T**: Leave commented-out code
- ❌ **DON'T**: Use magic numbers
- ❌ **DON'T**: Create overly complex code

### Testing

All code should include tests:

```python
import pytest
from unittest.mock import Mock

def test_rag_query():
    """Test RAG query with mocked LLM"""
    # Mock external dependencies
    mock_llm = Mock()
    mock_llm.predict.return_value = "Mocked answer"
    
    # Test your logic
    rag_system = RAGSystem(llm=mock_llm)
    result = rag_system.query("test question")
    
    # Assertions
    assert result == "Mocked answer"
    mock_llm.predict.assert_called_once()
```

Run tests before committing:
```bash
pytest tests/
```

### Documentation

Update documentation for:
- New features
- API changes
- Configuration changes
- Breaking changes

Add docstrings:
```python
def create_rag_chain(
    llm: BaseLLM,
    retriever: BaseRetriever,
    prompt_template: str = None
) -> RetrievalQA:
    """
    Create a RAG chain with best practices.
    
    Args:
        llm: Language model for generation
        retriever: Document retriever
        prompt_template: Optional custom prompt template
        
    Returns:
        Configured RetrievalQA chain
        
    Example:
        >>> llm = ChatOpenAI(temperature=0)
        >>> retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        >>> chain = create_rag_chain(llm, retriever)
    """
    ...
```

## 🔄 Commit Guidelines

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(langchain): add RAG chain with hybrid retrieval

Implemented RAG chain combining semantic and keyword search.
Uses MMR for diversity and re-ranking for quality.

Closes #42
```

```
fix(autogen): prevent infinite agent loops

Added max_iterations and max_execution_time to all agents.

Fixes #37
```

```
docs(rag): update chunking best practices

Added guidelines for different content types and token-based splitting.
```

## 📤 Submitting a Pull Request

### 1. Update Your Branch

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Push Your Changes

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

- Go to GitHub and create a pull request
- **Use the PR template** (it will auto-fill)
- Fill in ALL sections thoroughly
- Link related issues

### 4. Address Review Feedback

#### Automated Review
The AI Systems Architect bot will automatically review your PR:
- ✅ Fix all **critical issues** (blocking)
- ⚠️ Address **important issues** (strongly recommended)
- 💡 Consider **suggestions** (optional but beneficial)

#### Human Review
After addressing automated feedback:
- Request review from maintainers
- Respond to comments
- Make requested changes
- Be open to feedback

### 5. Merge Requirements

Your PR needs:
- ✅ All CI checks passing
- ✅ No critical security issues
- ✅ No merge conflicts
- ✅ At least one approval
- ✅ All conversations resolved

## 🏗️ Project Structure

When adding new code, follow this structure:

```
AdvancedAI/
├── month1-langchain/
│   ├── 01-basics/
│   ├── 02-chains/
│   ├── 03-agents/
│   └── 04-rag/
├── month2-multiagent/
│   ├── 01-autogen/
│   ├── 02-crewai/
│   └── 03-integration/
├── month3-production/
│   ├── 01-llamaindex/
│   ├── 02-semantic-kernel/
│   └── 03-deployment/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   ├── review-guidelines/
│   └── tutorials/
└── utils/
    ├── monitoring.py
    ├── security.py
    └── evaluation.py
```

## 🎯 What to Contribute

### High Priority
- RAG implementations with best practices
- Multi-agent system examples
- Production deployment guides
- Security improvements
- Evaluation frameworks
- Monitoring solutions

### Medium Priority
- Additional framework integrations
- Performance optimizations
- Documentation improvements
- Tutorial notebooks
- Testing utilities

### Nice to Have
- Example applications
- Benchmarking tools
- Cost optimization strategies
- UI/Dashboard components

## ❓ Questions?

- **Technical questions**: Open an issue with the `question` label
- **Bugs**: Open an issue with the `bug` label
- **Feature requests**: Open an issue with the `enhancement` label
- **Security issues**: Email directly (don't open public issues)

## 📜 Code of Conduct

Be respectful, inclusive, and professional. We're all learning.

## 🙏 Thank You!

Your contributions help build a valuable learning resource for the AI community!

---

**Remember**: Quality over quantity. One well-architected contribution is worth more than ten quick hacks.
