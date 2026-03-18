# Trust-Me-I-Know-This-Function

### Hijacking LLM Static Analysis using Bias (NDSS 2026)

🚨 **First demonstration of Familiar Pattern Attacks (FPAs)** — a new class of adversarial examples that exploit abstraction bias in code-understanding LLMs.

---

## 📌 Overview

Large Language Models (LLMs) are increasingly used for:
- Static code analysis  
- Vulnerability detection  
- Code summarization  

However, they suffer from a critical weakness:

> **Abstraction Bias** — the tendency to overgeneralize familiar code patterns and ignore small but meaningful deviations.

This repository contains the implementation of **Familiar Pattern Attacks (FPAs)**, which exploit this bias to **manipulate an LLM’s interpretation of code without changing runtime behavior**.

📄 Based on our paper:  
**"Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias" (NDSS 2026)**

---

## ⚡ Key Idea

An FPA works by:

1. Selecting a **familiar code pattern** (e.g., common algorithms)
2. Injecting a **subtle perturbation**
3. Embedding it into a program to **control what the LLM *believes*** happens

➡️ Result:
- The **code executes correctly**
- The **LLM confidently misinterprets it**

This allows attackers to:
- Hide vulnerabilities  
- Mislead code analysis  
- Corrupt summaries  

📖 As shown in our work, LLMs often rely on pattern recognition instead of precise reasoning, leading to systematic errors. :contentReference[oaicite:0]{index=0}

---

## 🧪 What’s in This Repository

- 🧠 FPA generation pipeline  
- 🔍 Evaluation scripts for LLM-based code analysis  
- 📊 Experiments on multiple models (e.g., GPT, Claude, Gemini)  
- 🌍 Cross-language examples (Python, C, etc.)  
- ⚙️ Reproducible setups for white-box & black-box attacks  

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/ShirBernBGU/Trust-Me-I-Know-This-Function.git
cd Trust-Me-I-Know-This-Function
