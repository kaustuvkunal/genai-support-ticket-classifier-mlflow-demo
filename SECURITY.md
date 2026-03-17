# Security Guide

## 🔐 Overview

This document outlines the security measures in place to protect API keys and sensitive information in this project.

## ✅ Security Measures Implemented

### 1. Environment Variables Protection

- **API keys are stored in `.env` file** (not in version control)
- **`.env` is listed in `.gitignore`** to prevent accidental commits
- **Placeholder values** are provided in `.env.example` for reference

### 2. Git Protection

- `.env` file is protected by `.gitignore` (line 138)
- `.envrc` (direnv files) are also ignored
- Configuration files with sensitive data are never committed

### 3. Logging Protection

The project uses safe logging practices:

- **No API keys are logged** - Only log initialization messages like "Initialized Groq client"
- **No environment variables are printed** - Configuration is loaded silently
- **Debug logs are sanitized** - Only log model names and safe parameters, never keys

Audit results:
- ✅ `logging_config.py` - Safe logging configuration
- ✅ `llm_client.py` - Initializes clients without logging API keys
- ✅ `config.py` - Loads configuration silently
- ✅ All other modules - No sensitive data in logs

### 4. Supported Providers

All 5 LLM providers have secure client implementations:

| Provider | Environment Variable | Example |
|----------|---------------------|---------|
| **Groq** | `GROQ_API_KEY` | `gsk_your_key_here` |
| **OpenAI** | `OPENAI_API_KEY` | `sk-proj-your_key_here` |
| **OpenAI GL** | `OPENAI_GL_API_KEY` | `gl-your_key_here` |
| **Azure OpenAI** | `AZURE_OPENAI_API_KEY` | `your_key_here` |
| **Google Gemini** | `GOOGLE_API_KEY` | `your_key_here` |

## 🚀 Setup Instructions

### Step 1: Copy Environment Template

```bash
# The .env file is not in git, so you need to create it
cp .env.example .env
```

### Step 2: Add Your API Keys

Edit `.env` and replace placeholder values with your real API keys:

```bash
# Example .env file with real values (DO NOT COMMIT)
GROQ_API_KEY=gsk_your_actual_groq_key_here
OPENAI_API_KEY=sk-proj-your_actual_openai_key_here
OPENAI_GL_API_KEY=gl-your_actual_greatlearning_key_here
```

### Step 3: Never Commit .env

```bash
# Verify .env is in .gitignore (it is)
grep ".env" .gitignore

# Check git status to ensure .env is not tracked
git status
```

## ⚠️ Best Practices

### DO ✅

- ✅ Store all API keys in `.env` file
- ✅ Use environment variables to load keys: `os.getenv("GROQ_API_KEY")`
- ✅ Log only safe information (model names, endpoints)
- ✅ Rotate keys regularly
- ✅ Use different keys for development vs. production

### DON'T ❌

- ❌ Never hardcode API keys in source code
- ❌ Never log API keys or environment variables
- ❌ Never commit `.env` file to version control
- ❌ Never share API keys in pull requests or issues
- ❌ Never leave keys in git history

## 🔄 If You Accidentally Expose a Key

### Immediate Actions

1. **Revoke the exposed key** in your provider's console
   - Groq: https://console.groq.com/keys
   - OpenAI: https://platform.openai.com/api-keys
   - Azure: Azure Portal → Cognitive Services
   - Google: Google Cloud Console

2. **Create a new key** and update `.env`

3. **Check git history** for any commits containing the key:
   ```bash
   git log --all -p | grep "your_old_key"
   ```

4. **If found in git, remove from history** using:
   ```bash
   # Option 1: BFG Repo-Cleaner (easier)
   bfg --replace-text passwords.txt --no-blob-protection .

   # Option 2: git filter-branch (harder but built-in)
   git filter-branch --tree-filter 'sed -i "s/old_key/new_key/g"' HEAD
   ```

## 📋 Verification Checklist

- [x] `.env` file exists and contains placeholder values
- [x] `.env` is in `.gitignore`
- [x] No API keys in `.env.example`
- [x] No logging of API keys in code
- [x] All LLM clients read keys from environment variables
- [x] Git history cleaned of exposed keys
- [x] SECURITY.md documentation in place

## 📚 References

- [OWASP: Secrets Management](https://owasp.org/www-community/controls/Secrets_Management)
- [Groq API Documentation](https://console.groq.com/docs)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Google Gemini API Documentation](https://ai.google.dev/)

## 🆘 Questions or Issues?

If you discover a security vulnerability, please keep it confidential and avoid disclosing in public issues. Follow responsible disclosure practices.
