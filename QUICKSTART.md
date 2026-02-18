# Quick Start Guide

Get started with the Advanced AI Systems Lab in 5 minutes!

## 🚀 Quick Setup

### 1. Clone the Repository
```bash
git clone https://github.com/cdHarshita/AdvancedAI.git
cd AdvancedAI
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys
```bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your API keys
# NEVER commit this file!
nano .env  # or use your favorite editor
```

Add at minimum:
```bash
OPENAI_API_KEY=your-key-here
```

### 4. Run the Example
```bash
# Try the good RAG example
python examples/good_rag_example.py
```

## 📚 What's Next?

### Learn the Framework
1. **Read the guides**:
   - [Architectural Review](docs/review-guidelines/ARCHITECTURAL_REVIEW.md)
   - [Security Best Practices](docs/review-guidelines/SECURITY_REVIEW.md)
   - [RAG Patterns](docs/review-guidelines/RAG_REVIEW.md)

2. **Study the examples**:
   - `examples/good_rag_example.py` - Production patterns
   - `examples/bad_rag_example.py` - What NOT to do

3. **Review the checklist**:
   - `.github/PULL_REQUEST_TEMPLATE/ai_systems_pr_template.md`

### Make Your First Contribution

1. **Create a branch**:
   ```bash
   git checkout -b feature/my-first-feature
   ```

2. **Add your code** (following the guidelines!)

3. **Test locally**:
   ```bash
   pytest tests/  # if you added tests
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "feat: add my feature"
   git push origin feature/my-first-feature
   ```

5. **Create PR** using the template

6. **Address automated review feedback**

## 🎯 Learning Path

### Month 1: LangChain & RAG (Weeks 1-4)
**Week 1**: Basics
- LangChain fundamentals
- Prompt templates
- Basic chains

**Week 2**: Advanced Chains
- Sequential chains
- Router chains
- Custom chains

**Week 3**: Agents & Tools
- Agent types
- Custom tools
- Tool integration

**Week 4**: RAG Implementation
- Document loading
- Chunking strategies
- Vector stores
- Complete RAG system

### Month 2: Multi-Agent Systems (Weeks 5-8)
**Week 5**: AutoGen Basics
- Agent setup
- Conversations
- Code execution

**Week 6**: AutoGen Advanced
- Multi-agent collaboration
- Human-in-the-loop
- Custom agents

**Week 7**: CrewAI
- Task orchestration
- Crew composition
- Tool integration

**Week 8**: Integration
- Combining frameworks
- Best practices
- Production patterns

### Month 3: Production Systems (Weeks 9-12)
**Week 9**: LlamaIndex
- Index types
- Query engines
- Advanced RAG

**Week 10**: Semantic Kernel
- Skills and planners
- Memory management
- Production patterns

**Week 11**: Deployment
- Docker containerization
- FastAPI services
- Cloud deployment

**Week 12**: Monitoring & Scaling
- Observability
- Performance tuning
- Cost optimization

## 🛠️ Development Workflow

### Daily Development
```bash
# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/your-feature

# Make changes
# ... edit files ...

# Test your changes
pytest tests/

# Commit
git add .
git commit -m "feat: description"

# Push and create PR
git push origin feature/your-feature
```

### Best Practices Checklist
Before every commit:
- [ ] No hardcoded API keys
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guide
- [ ] No security vulnerabilities
- [ ] Error handling added
- [ ] Logging implemented

## 📖 Resources

### Essential Reading
1. [README.md](README.md) - Project overview
2. [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
3. [Review Guidelines](docs/review-guidelines/) - Code review standards

### Framework Docs
- [LangChain](https://python.langchain.com/)
- [LlamaIndex](https://docs.llamaindex.ai/)
- [AutoGen](https://microsoft.github.io/autogen/)
- [CrewAI](https://docs.crewai.com/)

### Video Tutorials
- LangChain Official Channel
- DeepLearning.AI courses
- AutoGen tutorials

## 🆘 Getting Help

### Common Issues

**Issue**: `ModuleNotFoundError`
**Solution**: Activate venv and run `pip install -r requirements.txt`

**Issue**: `API key not found`
**Solution**: Check `.env` file exists and contains your API key

**Issue**: `Rate limit exceeded`
**Solution**: Check your API quota and add rate limiting

### Where to Ask
- **Bug reports**: Open an issue with `bug` label
- **Feature requests**: Open an issue with `enhancement` label
- **Questions**: Open an issue with `question` label
- **Security issues**: Email privately (see SECURITY.md)

## 🎉 First Project Ideas

Start with these beginner-friendly projects:

### 1. Document Q&A Bot
Build a RAG system that answers questions about your documents.

**Skills**: LangChain, embeddings, vector stores

### 2. Code Review Assistant
Create an agent that reviews code for best practices.

**Skills**: AutoGen, tool integration, prompt engineering

### 3. Research Assistant
Multi-agent system that researches topics and writes reports.

**Skills**: CrewAI, task orchestration, web scraping

### 4. Customer Support Bot
Conversational agent with memory and knowledge base.

**Skills**: LangChain, memory management, RAG

## 🏆 Success Criteria

You're on track when you can:
- ✅ Set up the dev environment independently
- ✅ Run the examples successfully
- ✅ Create a simple RAG system
- ✅ Write code that passes automated review
- ✅ Explain architectural decisions
- ✅ Identify security issues in code
- ✅ Contribute a meaningful PR

## 📈 Progress Tracking

Keep track of your learning:

```markdown
## My Progress

### Week 1
- [x] Environment setup
- [x] Ran first example
- [x] Read architectural guide
- [ ] Built first RAG system

### Week 2
- [ ] Advanced chains
- [ ] Custom tools
- [ ] First contribution
```

## 💡 Pro Tips

1. **Start small**: Don't try to build everything at once
2. **Read the guides**: They save hours of debugging
3. **Test frequently**: Catch issues early
4. **Ask questions**: The community is here to help
5. **Review others' code**: Learn from peers
6. **Iterate**: First version doesn't need to be perfect
7. **Monitor costs**: LLM calls add up quickly
8. **Version control**: Commit often, with clear messages

## 🎯 Next Steps

1. ✅ Complete the setup above
2. ✅ Run the examples
3. ✅ Read one review guide
4. ✅ Make your first contribution
5. ✅ Get your first PR merged!

---

**Welcome to Advanced AI Systems Lab!** 🚀

Let's build production-grade AI systems together.
