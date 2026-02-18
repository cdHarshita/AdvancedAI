# Testing the AI Architect Review System

This guide explains how to test the automated review system.

## 🧪 Overview

The review system consists of:
1. **Python-based static analysis** - Detects anti-patterns
2. **Security scanning** - Trivy for vulnerabilities
3. **Automated checklist** - Posted as PR comment

## 🔬 Local Testing

### Test the Python Analyzer

Run the analyzer on example files:

```bash
cd /home/runner/work/AdvancedAI/AdvancedAI

python3 << 'EOF'
import re
import sys
from pathlib import Path

issues = []
warnings = []
suggestions = []

# Scan Python files
for file_path in Path('.').rglob('*.py'):
    if '.venv' in str(file_path) or 'node_modules' in str(file_path):
        continue
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Security checks
        if re.search(r'api[_-]?key\s*=\s*["\'][^"\']+["\']', content, re.IGNORECASE):
            issues.append(f"❌ Hardcoded API key in `{file_path}`")
        
        # LangChain patterns
        if 'ConversationBufferMemory' in content and 'max_token_limit' not in content:
            warnings.append(f"⚠️ Unbounded memory in `{file_path}`")
        
        # RAG patterns
        if 'RecursiveCharacterTextSplitter' in content:
            if not re.search(r'chunk_size\s*=', content):
                issues.append(f"❌ Missing chunk_size in `{file_path}`")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Output
print(f"Found {len(issues)} issues, {len(warnings)} warnings, {len(suggestions)} suggestions")

if issues:
    print("\n❌ Critical Issues:")
    for issue in issues:
        print(f"  {issue}")

if warnings:
    print("\n⚠️ Warnings:")
    for warning in warnings:
        print(f"  {warning}")

sys.exit(1 if issues else 0)
EOF
```

### Expected Results

**On good code** (`examples/good_rag_example.py`):
```
Found 0 issues, 0 warnings, 0 suggestions
✅ All checks passed!
```

**On bad code** (`examples/bad_rag_example.py`):
```
Found 2 issues, 2 warnings, 1 suggestion
❌ Critical Issues:
  Hardcoded API key in examples/bad_rag_example.py
  Missing chunk_size in examples/bad_rag_example.py
⚠️ Warnings:
  Unbounded memory in examples/bad_rag_example.py
```

## 🎯 Test Cases

### Test Case 1: Hardcoded API Key Detection

**Create test file**:
```python
# test_security.py
from langchain.llms import ChatOpenAI

# This should be flagged
llm = ChatOpenAI(api_key="sk-proj-xxxxxxxxxxxx")
```

**Expected**: ❌ Critical issue detected

---

### Test Case 2: Unbounded Memory

**Create test file**:
```python
# test_memory.py
from langchain.memory import ConversationBufferMemory

# This should be flagged
memory = ConversationBufferMemory()
```

**Expected**: ⚠️ Warning issued

---

### Test Case 3: Missing Agent Guardrails

**Create test file**:
```python
# test_agent.py
from langchain.agents import initialize_agent

# This should be flagged
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```

**Expected**: ❌ Critical issue (no max_iterations)

---

### Test Case 4: Good Code (No Issues)

**Create test file**:
```python
# test_good.py
import os
from langchain.llms import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory

# This should pass
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000
)
```

**Expected**: ✅ No issues

## 🤖 Testing GitHub Actions Workflow

### Trigger the Workflow

The workflow triggers on:
- Pull request opened
- Pull request synchronized
- Pull request reopened

**Test by**:
1. Create a new branch
2. Make changes to Python files
3. Push to GitHub
4. Create a pull request

### What Gets Checked

1. **Changed files detection**
   - Only Python, Jupyter, Markdown files
   - Ignores non-code files

2. **Pattern analysis**
   - Hardcoded secrets
   - Framework anti-patterns
   - Missing best practices

3. **Security scan**
   - Trivy vulnerability scanner
   - Results uploaded to Security tab

4. **PR comment**
   - Comprehensive checklist posted
   - Includes review guidelines links

## 📊 Verification Checklist

After creating a PR, verify:

- [ ] Workflow runs automatically
- [ ] Python analysis executes
- [ ] Security scan completes
- [ ] PR comment is posted
- [ ] All checks are green (for good code)
- [ ] Issues are detected (for bad code)

## 🐛 Troubleshooting

### Workflow Doesn't Run

**Check**:
- `.github/workflows/ai-architect-review.yml` exists
- Workflow has correct triggers
- You have permissions to run workflows

**Fix**:
- Verify YAML syntax
- Check GitHub Actions settings
- Review workflow logs

### Python Analysis Fails

**Check**:
- Python 3.11 is available
- All imports work
- File paths are correct

**Fix**:
- Update Python version in workflow
- Add missing dependencies
- Fix file path references

### Security Scan Fails

**Check**:
- Trivy action is up to date
- SARIF upload has permissions

**Fix**:
- Update Trivy action version
- Add required permissions to workflow

### PR Comment Not Posted

**Check**:
- Workflow has `pull-requests: write` permission
- GitHub token is valid

**Fix**:
- Add permissions to workflow
- Use `${{ secrets.GITHUB_TOKEN }}`

## 🧩 Integration Tests

### Test Full PR Flow

1. **Create branch**:
   ```bash
   git checkout -b test/review-system
   ```

2. **Add bad code**:
   ```bash
   cp examples/bad_rag_example.py month1-langchain/test.py
   git add month1-langchain/test.py
   git commit -m "test: trigger review"
   git push origin test/review-system
   ```

3. **Create PR** on GitHub

4. **Verify**:
   - Workflow runs
   - Issues detected
   - Comment posted

5. **Fix issues**:
   ```bash
   cp examples/good_rag_example.py month1-langchain/test.py
   git add month1-langchain/test.py
   git commit -m "fix: address review feedback"
   git push origin test/review-system
   ```

6. **Verify**:
   - Workflow re-runs
   - No issues found
   - Checks pass

## 📈 Metrics to Track

Monitor these metrics over time:

### Code Quality
- **Issues per PR**: Should decrease
- **First-time pass rate**: Should increase
- **Review cycles**: Should decrease

### Security
- **Critical vulnerabilities**: Should be 0
- **Hardcoded secrets**: Should be 0
- **Security PRs blocked**: Track incidents

### Efficiency
- **Review time**: Automated review < 5 minutes
- **Human review time**: Should decrease
- **Time to merge**: Should decrease

## 🎓 Learning Exercises

### Exercise 1: Add New Check

Add detection for missing retry logic:

```python
# Add to workflow
if re.search(r'openai|anthropic', content, re.IGNORECASE):
    if not re.search(r'retry|tenacity', content, re.IGNORECASE):
        suggestions.append(f"💡 Consider retry logic in {file_path}")
```

### Exercise 2: Improve Detection

Make chunk size detection smarter:

```python
# Detect chunk size value
match = re.search(r'chunk_size\s*=\s*(\d+)', content)
if match:
    size = int(match.group(1))
    if size < 500 or size > 2000:
        warnings.append(f"⚠️ Unusual chunk_size {size} in {file_path}")
```

### Exercise 3: Custom Framework

Add checks for a new framework:

```python
# Semantic Kernel checks
if 'semantic_kernel' in content.lower():
    if 'Kernel' in content and not 'config' in content.lower():
        warnings.append(f"⚠️ Kernel without config in {file_path}")
```

## ✅ Success Criteria

The review system is working when:

1. **Detection**
   - ✅ Catches hardcoded secrets
   - ✅ Identifies anti-patterns
   - ✅ Suggests improvements

2. **Automation**
   - ✅ Runs on every PR
   - ✅ Completes in < 5 minutes
   - ✅ Posts helpful feedback

3. **Quality**
   - ✅ Low false positive rate
   - ✅ Clear, actionable feedback
   - ✅ Links to documentation

4. **Adoption**
   - ✅ Contributors use it
   - ✅ Code quality improves
   - ✅ Review time decreases

## 🚀 Continuous Improvement

### Monthly Review

1. **Analyze metrics**
   - Issues detected
   - False positives
   - Missed issues

2. **Update patterns**
   - Add new checks
   - Improve existing ones
   - Remove outdated checks

3. **Update documentation**
   - New examples
   - Clearer guidelines
   - Additional resources

### Community Feedback

Collect feedback on:
- Usefulness of checks
- Clarity of messages
- Actionability of suggestions
- Documentation quality

---

## 📞 Support

**Issues with the review system?**
- Open an issue with `review-system` label
- Include workflow logs
- Describe expected vs. actual behavior

**Want to improve the system?**
- Read the implementation in `.github/workflows/`
- Propose changes via PR
- Discuss in issues first for big changes

---

**Happy testing!** 🧪
