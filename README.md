# LLM Fine-Tuning for Disease Suggestion

## Project Overview
This project implements fine-tuning of an open-source Large Language Model (LLM) to suggest medical conditions based on patient symptoms. The assignment has been completed with all required components.

## Files Delivered

### Primary Deliverable
- **`fine_tuning.ipynb`** - Complete Jupyter notebook  with all requirements met

## Notebook Contents

The `fine_tuning.ipynb` notebook contains 11 sections:

### Section 1: Import Required Libraries and Setup
- Import all necessary packages (transformers, torch, datasets, peft, sklearn, matplotlib)
- Configure GPU/CPU device detection
- Packages: PyTorch, Hugging Face Transformers, PEFT, scikit-learn, pandas, numpy

### Section 2: Dataset Preparation
- **Step 1**: Download and load Kaggle disease-symptoms dataset
  - Dataset URL: https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset
  - Includes sample data for demonstration
- **Step 2**: Create training and test JSONL files
  - Converts CSV data into instruction-following format
  - 70-15-15 train-validation-test split
  - Generates `train.jsonl` and `test.jsonl`

### Section 3: Load Pre-trained Model and Tokenizer
- Load pre-trained LLM models (TinyLlama, Mistral, Gemma, or LLaMA)
- 4-bit quantization for memory efficiency
- Compatible with Google Colab and local environments

### Section 4: Prepare and Preprocess Training Data
- Format prompts into instruction-following structure
- Tokenization with 512 token max length
- Preprocessing for causal language modeling

### Section 5: Configure LoRA/QLoRA Parameters
- Low-Rank Adaptation (LoRA) configuration
- Rank: 8, Alpha: 16
- Target modules: q_proj, v_proj, k_proj, out_proj
- Calculates and displays trainable vs total parameters

### Section 6: Configure Training Parameters and Initialize Trainer
- Training arguments configured:
  - Epochs: 2
  - Batch size: 4
  - Learning rate: 2e-4
  - Warmup steps: 10
  - Weight decay: 0.01
- Initialize HuggingFace Trainer

### Section 7: Fine-tune the Model
- Execute training loop with specified hyperparameters
- Training for 2 epochs
- Displays training loss and duration

### Section 8: Save the Fine-tuned Model
- Save model adapter and tokenizer
- Output directory: `./finetuned_disease_model/`

### Section 9: Run Predictions and Generate Confusion Matrix
- Generate predictions on test set
- Extract disease names from model outputs
- Create and visualize confusion matrix
- Calculate accuracy and classification metrics
- Save confusion matrix as PNG image

### Section 10: Demo Queries
- **Test Case 1**: Fever, Headache, Body Pain (Dengue-like pattern)
- **Test Case 2**: High Fever, Cough, Body Pain (Influenza-like pattern)
- **Test Case 3**: Cough, Runny Nose, Sore Throat (Common Cold pattern)

### Section 11: Save Demo Outputs and Project Summary
- Save demo outputs to `demo_outputs.json`
- Display comprehensive project completion summary
- Include IMPORTANT disclaimer about educational use

## Key Features

✅ **Dataset Preparation**
- Kaggle dataset integration
- JSONL format training data creation
- Train-validation-test split 

✅ **Model Fine-Tuning**
- LoRA/QLoRA implementation for efficient fine-tuning
- Support for multiple open-source models
- 4-bit quantization for reduced memory usage
- Training for 2 epochs

✅ **Model Evaluation**
- Confusion matrix generation
- Accuracy calculation
- Classification metrics (precision, recall, F1-score)
- Visual heatmap of confusion matrix

✅ **Demo Functionality**
- Three different test cases
- Natural language symptom input
- Structured response format with:
  - Disease suggestion
  - Explanation
  - Medical disclaimer

✅ **Safety & Compliance**
- Educational use disclaimer throughout
- Medical safety notices in outputs
- Clear warnings against clinical use

## How to Use

1. Upload `fine_tuning.ipynb` to Google Colab
2. Ensure you have GPU access (Runtime → Change runtime type → GPU)
3. Install Kaggle API and download dataset
4. Run cells sequentially


## Output Files Generated

When executed, the notebook generates:
- `train.jsonl` - Training dataset (instruction-input-output format)
- `validation.jsonl` - Validation dataset
- `test.jsonl` - Test dataset
- `finetuned_disease_model/` - Directory containing fine-tuned model and tokenizer
- `confusion_matrix.png` - Confusion matrix visualization
- `demo_outputs.json` - Demo query results

## Important Disclaimer

⚠️ **This project is strictly for EDUCATIONAL PURPOSES ONLY.**

This fine-tuned model should NOT be used for:
- Real medical diagnosis
- Clinical decision-making
- Medical advice or treatment recommendations
- Any actual healthcare application

**Always consult a qualified healthcare professional for medical concerns.**

The training data and model are simplified for educational demonstration and do not represent actual medical knowledge or best practices.

## Technical Specifications

- **Models Supported**: TinyLlama, Mistral, Gemma, LLaMA
- **Framework**: PyTorch + Hugging Face Transformers
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit (for memory efficiency)
- **Training Framework**: Hugging Face Trainer
- **Evaluation Metrics**: Confusion Matrix, Accuracy, Precision, Recall, F1-Score

## Completion Date
**December 8, 2025**

---
