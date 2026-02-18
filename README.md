# 🚀 Advanced AI Systems Lab

**A 3-Month Structured Learning Path for Production-Grade AI Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Review](https://github.com/cdHarshita/AdvancedAI/workflows/AI%20Systems%20Architect%20Review/badge.svg)](https://github.com/cdHarshita/AdvancedAI/actions)

## 📋 Overview

This repository is a comprehensive 90-day Advanced AI Systems Lab designed to build enterprise-grade AI applications using modern frameworks and best practices. Every contribution is reviewed by our **Senior AI Systems Architect** automated review system to ensure production-quality code.

## 🎯 Learning Curriculum

### Month 1: LangChain Foundations & RAG
- ✅ LangChain fundamentals
- ✅ Prompt engineering best practices
- ✅ Chains and advanced chains
- ✅ Agents and custom tools
- ✅ Embeddings and vector stores
- ✅ Full RAG implementation

### Month 2: Multi-Agent Systems
- ✅ Multi-agent systems using AutoGen
- ✅ CrewAI task orchestration
- ✅ Nested and hierarchical agents
- ✅ Tool integration and execution
- ✅ Hybrid framework integrations

### Month 3: Production Systems
- ✅ Advanced RAG using LlamaIndex
- ✅ Multi-index and hybrid retrieval
- ✅ Agents inside LlamaIndex
- ✅ Production systems using Semantic Kernel
- ✅ Deployment using Docker + FastAPI
- ✅ Cloud deployment (Azure/AWS/GCP)

## 🏗️ Architectural Review System

This repository features an **automated Senior AI Systems Architect** that reviews every pull request for:

### 1. **Architectural Correctness** ✓
- Validates design patterns
- Ensures separation of concerns
- Identifies better alternatives
- Flags over-engineering

### 2. **Framework Best Practices** ✓
- LangChain patterns and anti-patterns
- LlamaIndex configuration
- AutoGen multi-agent design
- CrewAI task orchestration
- Semantic Kernel production patterns

### 3. **Security** 🔒
- API key and secrets management
- Input validation
- Prompt injection protection
- PII detection and handling
- Rate limiting

### 4. **RAG Quality** 📚
- Chunking strategy validation
- Embedding model selection
- Retrieval optimization
- Hallucination mitigation
- Evaluation metrics

### 5. **Production Readiness** 🚀
- Scalability assessment
- Error handling & resilience
- Cost optimization
- Monitoring & observability
- Performance metrics

## 📖 Documentation

### Review Guidelines
Comprehensive guides for reviewing AI systems:

- **[Architectural Review Guide](docs/review-guidelines/ARCHITECTURAL_REVIEW.md)** - Best practices for LangChain, LlamaIndex, AutoGen, CrewAI, and Semantic Kernel
- **[Security Review Guide](docs/review-guidelines/SECURITY_REVIEW.md)** - Security best practices for AI systems (OWASP LLM Top 10)
- **[RAG Review Guide](docs/review-guidelines/RAG_REVIEW.md)** - Comprehensive RAG implementation patterns

### PR Template
Use our [AI Systems PR Template](.github/PULL_REQUEST_TEMPLATE/ai_systems_pr_template.md) for structured, comprehensive pull requests.

## 🔍 How It Works

### Automated Reviews
Every pull request triggers:

1. **Architectural Analysis** - Python static analysis for common AI anti-patterns
2. **Security Scan** - Trivy vulnerability scanner
3. **Automated Checklist** - Comprehensive review checklist posted as PR comment
4. **Best Practice Validation** - Framework-specific pattern checking

### What Gets Reviewed

#### Code Patterns Detected
- ❌ Hardcoded API keys
- ❌ Unbounded memory buffers
- ❌ Missing agent guardrails
- ❌ Inefficient RAG chunking
- ❌ Missing retry logic
- ❌ Prompt injection vulnerabilities

#### Suggestions Provided
- 💡 Callback handlers for monitoring
- 💡 Token limits and cost management
- 💡 Better retrieval strategies
- 💡 Error handling improvements
- 💡 Performance optimizations

## 🚦 Getting Started

### Prerequisites
- Python 3.9+
- Git
- GitHub account

### Setup
```bash
# Clone the repository
git clone https://github.com/cdHarshita/AdvancedAI.git
cd AdvancedAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (as you add them)
pip install -r requirements.txt

# Create .env file (never commit this!)
cp .env.example .env
# Add your API keys to .env
```

### Making Your First Contribution

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes following best practices**
   - See review guidelines in `docs/review-guidelines/`
   - Use the PR template
   - Add tests

3. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: your descriptive message"
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**
   - Use the AI Systems PR template
   - Fill in all sections
   - Wait for automated review

5. **Address Review Comments**
   - Review automated feedback
   - Make necessary changes
   - Request human review if needed

## 📁 Repository Structure

```
AdvancedAI/
├── .github/
│   ├── workflows/
│   │   └── ai-architect-review.yml    # Automated review workflow
│   └── PULL_REQUEST_TEMPLATE/
│       └── ai_systems_pr_template.md  # PR template
├── docs/
│   └── review-guidelines/
│       ├── ARCHITECTURAL_REVIEW.md    # Architecture best practices
│       ├── SECURITY_REVIEW.md         # Security guidelines
│       └── RAG_REVIEW.md              # RAG implementation guide
├── month1-langchain/                  # Month 1 projects
├── month2-multiagent/                 # Month 2 projects
├── month3-production/                 # Month 3 projects
├── .env.example                       # Environment variables template
├── .gitignore
└── README.md
```

## 🎓 Learning Resources

### Official Documentation
- [LangChain Docs](https://python.langchain.com/)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [AutoGen Docs](https://microsoft.github.io/autogen/)
- [CrewAI Docs](https://docs.crewai.com/)
- [Semantic Kernel Docs](https://learn.microsoft.com/en-us/semantic-kernel/)

### Best Practices
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [LangChain Production Guide](https://python.langchain.com/docs/guides/productionization/)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

## 🛡️ Security

This project follows security best practices:

- ✅ No hardcoded secrets
- ✅ Environment-based configuration
- ✅ Input validation
- ✅ Output sanitization
- ✅ Rate limiting
- ✅ PII protection

**Never commit:**
- API keys
- Passwords
- `.env` files
- Sensitive data

## 🤝 Contributing

We welcome contributions! Please:

1. Read the [Architectural Review Guide](docs/review-guidelines/ARCHITECTURAL_REVIEW.md)
2. Follow the [Security Guidelines](docs/review-guidelines/SECURITY_REVIEW.md)
3. Use the PR template
4. Write tests
5. Address automated review feedback

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This project uses:
- LangChain for LLM orchestration
- LlamaIndex for advanced RAG
- AutoGen for multi-agent systems
- CrewAI for task orchestration
- Semantic Kernel for production patterns

## 📞 Contact

For questions or suggestions, please open an issue.

---

**Built with ❤️ for learning production-grade AI systems**
