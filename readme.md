# 📸 UNC Charlotte Image Generation Project

### *Fine-Tuned Stable Diffusion 1.5 & Stable Diffusion 3 for Location-Conditioned Image Generation*

This repository contains the implementation for a UNC Charlotte–themed image generation system built using **Stable Diffusion 1.5 (SD1.5)** and **Stable Diffusion 3 (SD3)** models. Both base and fine-tuned versions are supported, allowing reproducible generation of campus-specific images conditioned on locations and optional scene attributes.

A **Streamlit** web application (`app.py`) is included to make the image generation interface easy to use.

---

## 🚀 Features

* **Fine-tuned SD1.5 & SD3 models** on UNC Charlotte campus dataset
* Generate images conditioned on:

  * Campus *location tokens*
  * Optional *scene attributes* (trees, grass, road, water, buildings, etc.)
* Clean, modular codebase for:

  * Dataset preparation
  * Deduplication
  * Model building/training
  * Inference pipelines
  * Image correction
* Fully reproducible workflow with clearly structured folders

---

## 📁 Repository Structure

The folder organization in this repository is designed to be **simple, modular, and self-explanatory**, enabling anyone to understand or reproduce the results without guesswork:

```
├── Dataset/
├── Deduplication/
├── Deepseek/DeepSeek-VL2/
├── Image_Correction/
├── SD3_finetune/
├── SD_dataset_preparation/
├── SD_finetune/
├── app.py
└── readme.md
```

Each directory contains logically grouped components—data processing scripts, training code, models, evaluation utilities, and inference logic—making experimentation or extension straightforward.

---

## 📦 Checkpoints & Dataset

All model checkpoints, training logs, and the curated dataset used for training are stored in Google Drive:

🔗 **Drive Link (Checkpoints & Dataset):**
[https://drive.google.com/drive/folders/1Fynn8j6JSFnOlPDzsD_6vEftwlxYuCdw?usp=sharing](https://drive.google.com/drive/folders/1Fynn8j6JSFnOlPDzsD_6vEftwlxYuCdw?usp=sharing)

Place them inside the expected directory paths before running the app or inference scripts.

---

## ▶️ Running the Streamlit App

### **1. Install Dependencies**

### **2. Run the Application**

Inside the project root:

```bash
streamlit run app.py
```

This launches the browser interface where you can:

* Choose campus locations
* Select model (base or fine-tuned)
* Add optional attributes
* Generate images with one click

The app automatically loads:

* Location token mappings
* SD1.5 / SD3 base pipelines
* Fine-tuned checkpoints (from the Drive assets)

---

## 📐 How It Works

### **Location Tokens**

Custom location-specific tokens were created and mapped to dataset images, enabling conditioning during training and inference.

### **Fine-Tuning**

Both SD1.5 and SD3 models were fine-tuned on high-quality UNC Charlotte campus images with:

* Caption rewriting
* Deduplication
* Image correction
* Data balancing
* Location-token injection

### **Inference**

The system constructs prompts like:

```
a photo of <LOCATION_TOKEN> <noun> at UNC Charlotte, trees, road, water
```

These are passed to the appropriate pipeline depending on your model selection.

---

## 🛠️ Reproducibility

This repository is intentionally structured to allow **full reproduction** of the experiments:

* All training scripts and configs are included
* Folder names correspond directly to pipeline stages
* Datasets and checkpoints are available via Drive
* The Streamlit app runs directly with no additional modification

---
