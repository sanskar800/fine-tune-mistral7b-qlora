# Fine-Tuning Mistral-7B-Instruct with QLoRA on Google Colab (T4)

This repository demonstrates how to fine-tune **mistralai/Mistral-7B-Instruct-v0.2** using **QLoRA (4-bit)** on a **Google Colab T4 GPU**, with the **Databricks Dolly 15k** dataset.  
The workflow covers **training, saving LoRA adapters, and running inference** with the fine-tuned model.

---

## 🚀 Features

- ✅ 4-bit QLoRA fine-tuning with **bitsandbytes**
- ✅ Optimized for **Google Colab T4 (16 GB VRAM)**
- ✅ Uses **TRL `SFTTrainer`**
- ✅ Instruction tuning in **Mistral-Instruct format**
- ✅ Adapter-based training (no full model weights saved)
- ✅ Simple inference function included

---

## 📦 Model & Dataset

- **Base Model:**  
  [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

- **Dataset:**  
  [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

---

## 🧠 Training Method

- **Technique:** QLoRA (4-bit)
- **Quantization:** NF4 + Double Quantization
- **LoRA Target Modules:**
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `gate_proj`, `up_proj`, `down_proj`

---

## 🖥️ Environment

Tested on:

- Google Colab
- GPU: **NVIDIA T4 (16 GB VRAM)**
- CUDA enabled

---

## 🔧 Installation

```bash
pip install -U pip
pip install -U \
  "transformers>=4.46.0" \
  "accelerate>=0.34.0" \
  "peft>=0.13.0" \
  "trl==0.26.0" \
  "datasets>=3.0.0" \
  "bitsandbytes>=0.43.0"

pip install -q triton
```
