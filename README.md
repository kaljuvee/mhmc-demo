# MCMC & Bayesian Inference Interactive Demo

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)
![License](https://img.shields.io/badge/license-MIT-green)

An interactive web application demonstrating Metropolis-Hastings Monte Carlo (MCMC) sampling and Bayesian Linear Regression using Streamlit and Plotly.

## 📋 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### 1. Metropolis-Hastings Sampling Demo
- Interactive sampling from a mixture of Gaussian distributions
- Adjustable parameters:
  - Number of samples
  - Proposal distribution width
  - Initial value
- Real-time visualization of:
  - Sampling results
  - True distribution
  - Acceptance rate

### 2. Bayesian Linear Regression Demo
- Interactive Bayesian regression with synthetic data
- Configurable parameters:
  - Number of data points
  - Noise level
  - MCMC samples
  - Random seed
- Visualizations:
  - Data points with regression lines
  - Parameter posterior distributions
  - Uncertainty quantification

### 3. Interactive Visualizations
- Powered by Plotly
- Features:
  - Zoom functionality
  - Pan controls
  - Hover information
  - Download options
  - Reset view

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/mcmc-demo.git
cd mcmc-demo
```

### Step 2: Create a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the App
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Using the Demo

#### Metropolis-Hastings Sampling
1. Select "Metropolis-Hastings Sampling" from the sidebar
2. Adjust the parameters:
   - Number of Samples (1,000 - 50,000)
   - Proposal Width (0.1 - 2.0)
   - Initial Value
3. Click "Run Sampling"
4. Observe the results in the interactive plot

#### Bayesian Linear Regression
1. Select "Bayesian Linear Regression" from the sidebar
2. Configure the settings:
   - Number of Data Points (50 - 500)
   - Noise Level (0.1 - 2.0)
   - MCMC Samples (1,000 - 20,000)
   - Random Seed
3. Click "Run Bayesian Regression"
4. Examine the results:
   - Regression plot with uncertainty
   - Parameter posterior distributions
   - Parameter estimates with uncertainties

## 🔧 Technical Details

### Implementation Overview

#### Metropolis-Hastings Algorithm
```python
class MetropolisHastings:
    def __init__(self, target_distribution, proposal_width=1.0):
        self.target_distribution = target_distribution
        self.proposal_width = proposal_width
    
    def sample(self, n_samples, initial_value):
        # Implementation details in app.py
```

#### Bayesian Linear Regression
```python
class BayesianLinearRegression:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_samples, self.n_features = self.X.shape
    
    def sample_posterior(self, n_samples, proposal_width=0.1):
        # Implementation details in app.py
```

### Key Components
1. **MCMC Sampling Engine**
   - Custom implementation of Metropolis-Hastings
   - Efficient numpy-based computations
   - Acceptance rate monitoring

2. **Bayesian Regression**
   - Linear regression with uncertainty quantification
   - Prior and likelihood calculations
   - MCMC-based posterior sampling

3. **Visualization Engine**
   - Plotly-based interactive plots
   - Real-time updates
   - Multiple visualization types

## 📁 Project Structure
```
mcmc-demo/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 📈 Examples

### Example 1: Simple Mixture Model
```python
# Using the MetropolisHastings class
mh = MetropolisHastings(target_log_pdf, proposal_width=0.5)
samples, acceptance_rate = mh.sample(n_samples=10000, initial_value=0.0)
```

### Example 2: Bayesian Regression
```python
# Generate synthetic data
X, y, true_beta = generate_synthetic_data(n_samples=100, noise_std=0.5)

# Fit model
model = BayesianLinearRegression(X, y)
samples, acceptance_rate = model.sample_posterior(n_samples=5000)
```

## ❗ Troubleshooting

### Common Issues

1. **ImportError: No module named 'streamlit'**
   - Solution: Ensure you've activated the virtual environment and installed requirements
   ```bash
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Low Acceptance Rate**
   - Solution: Try reducing the proposal width
   - Optimal acceptance rate is typically between 20% and 50%

3. **Slow Performance**
   - Solution: Reduce the number of samples or data points
   - Consider using a more powerful machine for large sample sizes

### Performance Tips
1. Start with smaller sample sizes to test
2. Adjust proposal width for better acceptance rates
3. Use reasonable noise levels for synthetic data

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact
For questions and feedback:
- Create an issue in the repository
- Contact: your.email@example.com

## 🙏 Acknowledgments
- Streamlit team for the amazing framework
- Plotly team for the interactive visualization library
- The Bayesian statistics community for inspiration and methodologies
