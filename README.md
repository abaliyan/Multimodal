<h1>Multimodal Catalyst Performance Prediction</h1>
<hr><p>Project Name: Multimodal Catalyst</p>
<p>Performance Prediction</p>
<p>Version: 1.0.0</p>
<p>Author: A. Baliyan</p>
<p>License: MIT</p><h2>General Information</h2>
<hr><ul>
<li>Problem Statement: Traditional catalyst performance prediction relies on single characterization techniques, limiting accuracy and interpretability. This project addresses the challenge of integrating multiple spectroscopic modalities for enhanced prediction capabilities.</li>
</ul><ul>
<li>Solution Approach: Novel permutation strategy combined with advanced data augmentation techniques to overcome data scarcity issues while maintaining physical interpretability of machine learning models.</li>
</ul><ul>
<li>Impact: Enables accelerated catalyst discovery and design by providing reliable structure-performance relationships through multimodal analysis, reducing experimental time and costs in catalyst development.</li>
</ul><h2>Technologies Used</h2>
<hr><ul>
<li>Python 3.8+ - Core programming language</li>
</ul><ul>
<li>Scikit-learn - Machine learning algorithms (Linear Regression, Decision Tree, Random Forest)</li>
</ul><ul>
<li>XGBoost - Gradient boosting framework</li>
</ul><ul>
<li>NumPy &amp; Pandas - Data manipulation and numerical computing</li>
</ul><ul>
<li>Matplotlib &amp; Seaborn - Data visualization</li>
</ul><ul>
<li>SciPy - Scientific computing and statistical functions</li>
</ul><h2>Features</h2>
<hr><ul>
<li>Multimodal Data Integration: Seamlessly combines 8 spectroscopic techniques (EXAFS, XRD, XANES, PDF, HAXPS-VB, SAXS, HAXPS-Pt3d, HAXPS-Pt4f)</li>
</ul><ul>
<li>Advanced Preprocessing Pipeline: Uniform resampling to 300 data points and min-max normalization across all modalities</li>
</ul><ul>
<li>Novel Permutation Strategy: Systematic dataset augmentation creating 8 different configurations (Singlet to Octa)</li>
</ul><ul>
<li>Dual Data Augmentation: Gaussian Process with RBF kernel and spectral mix-up using Dirichlet distribution</li>
</ul><ul>
<li>Multiple ML Model Comparison: Comprehensive evaluation of 5 different algorithms with cross-validation</li>
</ul><ul>
<li>Interpretable AI: Feature importance analysis with physical interpretation requirements</li>
</ul><ul>
<li>Modality Ranking System: Quantitative assessment of individual modality contributions</li>
</ul><h2>Setup</h2>
<hr><p>Clone the Repository
[bash]</p>
<p>git clone https://github.com/abaliyan/Multimodal.git
cd Multimodal</p>
<p>Create Virtual Environment
[bash]</p>
<p>python -m venv multimodal_env
source multimodal_env/bin/activate  # On Windows: multimodal_env\Scripts\activate</p>
<p>Install Dependencies
[bash]</p>
<p>pip install -r requirements.txt</p>
<p>Run the script according to your data</p><h2>Project Status</h2>
<hr><p>Completed</p>
