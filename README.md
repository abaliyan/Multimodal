# Multimodal Catalyst Performance Prediction

---

**Project Name**: Multimodal Catalyst  
**Performance Prediction**  
**Version**: 1.0.0  
**Author**: A. Baliyan  
**License**: MIT

---

## General Information

- **Problem Statement**: Traditional catalyst performance prediction relies on single characterization techniques, limiting accuracy and interpretability. This project addresses the challenge of integrating multiple spectroscopic modalities for enhanced prediction capabilities.
- **Solution Approach**: Novel permutation strategy combined with advanced data augmentation techniques to overcome data scarcity issues while maintaining physical interpretability of machine learning models.
- **Impact**: Enables accelerated catalyst discovery and design by providing reliable structure-performance relationships through multimodal analysis, reducing experimental time and costs in catalyst development.

---

## Technologies Used

- **Python 3.8+** – Core programming language  
- **Scikit-learn** – ML algorithms (Linear Regression, Decision Tree, Random Forest)  
- **XGBoost** – Gradient boosting framework  
- **NumPy & Pandas** – Data manipulation and numerical computing  
- **Matplotlib & Seaborn** – Data visualization  
- **SciPy** – Scientific computing and statistical functions  

---

## Features

- **Multimodal Data Integration**: Combines 8 spectroscopic techniques (EXAFS, XRD, XANES, PDF, HAXPS-VB, SAXS, HAXPS-Pt3d, HAXPS-Pt4f)
- **Advanced Preprocessing Pipeline**: Uniform resampling to 300 data points and min-max normalization
- **Novel Permutation Strategy**: Augments data into 8 configurations (Singlet to Octa)
- **Dual Data Augmentation**: Uses Gaussian Process with RBF kernel and spectral mix-up with Dirichlet distribution
- **Multiple ML Model Comparison**: Evaluates 5 algorithms with cross-validation
- **Interpretable AI**: Feature importance analysis tied to physical meaning
- **Modality Ranking System**: Quantitative assessment of each modality's contribution

---

## Setup

**Clone the Repository**:

```bash
git clone https://github.com/abaliyan/Multimodal.git
cd Multimodal

Create a Virtual Environment:

python -m venv multimodal_env
source multimodal_env/bin/activate  # On Windows: multimodal_env\Scripts\activate

#Install Dependencies:

pip install -r requirements.txt
```
**Replace the Data in ["MM_dataset_II_III_IV_V_Aug2024"]**:
Run the scripts according to the need.

## Project Status
- Completed

