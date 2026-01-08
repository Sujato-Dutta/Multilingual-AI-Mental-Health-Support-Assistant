# 💚 Multilingual AI Mental Health Support Assistant

A production-grade, multilingual AI mental health support assistant with text capabilities, deployed via Streamlit Community Cloud.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **🌍 Multilingual Support**: Automatic language detection and translation
- **🛡️ Safety-First Design**: 3-tier risk classification with automatic escalation
- **💬 Conversational Memory**: Context-aware responses with bounded history
- **🤖 Fine-tuned Models**: LoRA-adapted models for mental health support
- **⚡ CPU-Only Execution**: No GPU required for inference

## 🏗️ Architecture

```
User Input (text)
    ↓
Language Detection (langdetect)
    ↓
Translate to English (argos-translate)
    ↓
Risk Classification (DistilRoBERTa + LoRA)
    ↓
├─ HIGH Risk → Predefined Escalation Response
└─ LOW/MEDIUM → Response Generation (Qwen2.5-0.5B + LoRA)
    ↓
Safety Guardrail Check
    ↓
Translate to Original Language
    ↓
Streamlit UI Output
```

## 📁 Project Structure

```
AI Mental Health Support Assistant/
├── app/                    # Streamlit application
│   ├── main.py            # Entry point
│   └── components/        # UI components
├── models/                 # Model wrappers
│   ├── risk_classifier.py
│   └── response_generator.py
├── training/               # Training scripts
│   ├── train_risk_classifier.py
│   ├── train_response_generator.py
│   └── train_safety_adapter.py
├── inference/              # Inference pipeline
│   └── pipeline.py
├── translation/            # Language services
│   └── translator.py
├── safety/                 # Safety guardrails
│   ├── guardrails.py
│   └── escalation.py
├── configs/                # Configuration
│   ├── model_config.py
│   └── prompts.py
├── utils/                  # Utilities
├── data/                   # Datasets
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 4GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-mental-health-assistant.git
cd ai-mental-health-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

## 🧠 Model Training (Optional)

Training requires significant compute resources. Pre-trained adapters can be used if available. The models will use keyword-based fallbacks until the LoRA adapters are trained. Training requires running the scripts in the training/ folder (CPU-intensive, may take hours).

### Train Risk Classifier

```bash
python training/train_risk_classifier.py
```

### Train Response Generator

```bash
python training/train_response_generator.py
```

### Train Safety Adapter

```bash
python training/train_safety_adapter.py
```

## 🌐 Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Connect repository to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Set entry point: `app/main.py`
4. Deploy!

## ⚠️ Safety Features

### Risk Classification

| Level | Description | Response |
|-------|-------------|----------|
| LOW | General emotional support | Generated response |
| MEDIUM | Elevated concern | Deterministic generation |
| HIGH | Crisis indicators | Predefined escalation |

### Safety Guardrails

- Pattern detection for unsafe content
- Automatic response override
- Crisis resource injection
- No medical/diagnostic advice

### Crisis Resources

- **988** - Suicide & Crisis Lifeline (US)
- **741741** - Crisis Text Line (text HOME)
- **911** - Emergency Services

## 📊 Technical Specifications

### Models

| Component | Model | Method |
|-----------|-------|--------|
| Response Generator | Qwen2.5-0.5B-Instruct | LoRA |
| Risk Classifier | DistilRoBERTa | LoRA |
| Translation | Argos Translate | - |

### LoRA Configuration

- LoRA rank: 8
- LoRA alpha: 16
- Target modules: q_proj, k_proj, v_proj, o_proj

## 🔧 Configuration

Environment variables (see `.env.example`):

```bash
LOG_LEVEL=INFO
DEBUG=false
```

## 📝 Important Disclaimers

> ⚠️ **This AI assistant is NOT a licensed therapist or medical professional.**

- Provides emotional support only
- Cannot diagnose conditions
- Cannot prescribe treatment
- Not a substitute for professional help

**If you're in crisis, please contact:**
- 988 (Suicide & Crisis Lifeline)
- 911 (Emergency)
- Your local emergency services

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- Hugging Face Transformers
- Argos Translate for multilingual support
- Streamlit for the web framework

---

Made with 💚 for mental health awareness
