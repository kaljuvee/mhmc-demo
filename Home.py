import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import pandas as pd

# Set page config
st.set_page_config(
    page_title="MCMC Demonstration",
    page_icon="📊",
    layout="wide"
)

# Define the MetropolisHastings class
class MetropolisHastings:
    def __init__(self, target_distribution, proposal_width=1.0):
        self.target_distribution = target_distribution
        self.proposal_width = proposal_width
        
    def sample(self, n_samples, initial_value):
        samples = np.zeros(n_samples)
        current = initial_value
        accepted = 0
        
        for i in range(n_samples):
            proposal = current + np.random.normal(0, self.proposal_width)
            log_ratio = (self.target_distribution(proposal) - 
                        self.target_distribution(current))
            
            if np.log(np.random.random()) < log_ratio:
                current = proposal
                accepted += 1
            
            samples[i] = current
            
        acceptance_rate = accepted / n_samples
        return samples, acceptance_rate

class BayesianLinearRegression:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_samples, self.n_features = self.X.shape
        
    def log_likelihood(self, beta, sigma):
        y_pred = self.X @ beta
        return np.sum(norm.logpdf(self.y, y_pred, sigma))
    
    def log_prior(self, beta, sigma):
        beta_prior = np.sum(norm.logpdf(beta, 0, 10))
        sigma_prior = norm.logpdf(sigma, 0, 10)
        return beta_prior + sigma_prior
    
    def log_posterior(self, params):
        beta = params[:-1]
        sigma = params[-1]
        if sigma <= 0:
            return -np.inf
        return self.log_likelihood(beta, sigma) + self.log_prior(beta, sigma)
    
    def propose(self, current, proposal_width):
        return np.random.multivariate_normal(
            current, 
            proposal_width * np.eye(len(current))
        )
    
    def sample_posterior(self, n_samples, proposal_width=0.1):
        current = np.zeros(self.n_features + 1)
        samples = np.zeros((n_samples, self.n_features + 1))
        accepted = 0
        
        for i in range(n_samples):
            proposal = self.propose(current, proposal_width)
            log_ratio = (self.log_posterior(proposal) - 
                        self.log_posterior(current))
            
            if np.log(np.random.random()) < log_ratio:
                current = proposal
                accepted += 1
            
            samples[i] = current
            
        acceptance_rate = accepted / n_samples
        return samples, acceptance_rate

def generate_synthetic_data(n_samples=100, noise_std=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.normal(0, 1, (n_samples, 2))
    true_beta = np.array([1.5, -0.8])
    y = X @ true_beta + np.random.normal(0, noise_std, n_samples)
    return X, y, true_beta

def target_log_pdf(x):
    """Log probability density function for a mixture of two Gaussians"""
    return np.log(0.5 * norm.pdf(x, -2, 1) + 0.5 * norm.pdf(x, 2, 1.5))

# Streamlit app starts here
st.title("📊 MCMC & Bayesian Inference Demonstration")

# Sidebar controls
st.sidebar.header("Configuration")

demo_type = st.sidebar.radio(
    "Select Demo Type",
    ["Metropolis-Hastings Sampling", "Bayesian Linear Regression"]
)

if demo_type == "Metropolis-Hastings Sampling":
    st.header("Metropolis-Hastings Sampling from Mixture of Gaussians")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        n_samples = st.slider(
            "Number of Samples",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000
        )
        
        proposal_width = st.slider(
            "Proposal Distribution Width",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1
        )

    with col2:
        initial_value = st.number_input(
            "Initial Value",
            value=0.0,
            step=0.1
        )
        
        if st.button("Run Sampling"):
            with st.spinner("Sampling in progress..."):
                # Run MCMC
                mh = MetropolisHastings(target_log_pdf, proposal_width)
                samples, acceptance_rate = mh.sample(n_samples, initial_value)
                
                # Create visualization
                true_x = np.linspace(-6, 6, 1000)
                true_density = np.exp([target_log_pdf(x) for x in true_x])
                
                # Create Plotly figure
                fig = go.Figure()
                
                # Add histogram of samples
                fig.add_trace(go.Histogram(
                    x=samples,
                    name="MCMC Samples",
                    histnorm='probability density',
                    nbinsx=50,
                    opacity=0.7
                ))
                
                # Add true distribution
                fig.add_trace(go.Scatter(
                    x=true_x,
                    y=true_density,
                    name="True Distribution",
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title="Metropolis-Hastings Sampling Results",
                    xaxis_title="x",
                    yaxis_title="Density",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"Acceptance Rate: {acceptance_rate:.2%}")

else:  # Bayesian Linear Regression
    st.header("Bayesian Linear Regression with MCMC")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        n_samples_data = st.slider(
            "Number of Data Points",
            min_value=50,
            max_value=500,
            value=100,
            step=50
        )
        
        noise_std = st.slider(
            "Noise Level",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1
        )

    with col2:
        mcmc_samples = st.slider(
            "MCMC Samples",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=1000
        )
        
        random_seed = st.number_input(
            "Random Seed",
            value=42,
            step=1
        )
        
    if st.button("Run Bayesian Regression"):
        with st.spinner("Running MCMC..."):
            # Generate synthetic data
            X, y, true_beta = generate_synthetic_data(
                n_samples=n_samples_data,
                noise_std=noise_std,
                seed=random_seed
            )
            
            # Fit model
            model = BayesianLinearRegression(X, y)
            samples, acceptance_rate = model.sample_posterior(
                n_samples=mcmc_samples,
                proposal_width=0.1
            )
            
            # Create visualizations
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Scatter plot with regression lines
                fig1 = go.Figure()
                
                # Add data points
                fig1.add_trace(go.Scatter(
                    x=X[:, 0],
                    y=y,
                    mode='markers',
                    name='Data Points',
                    marker=dict(size=8, opacity=0.6)
                ))
                
                # Add regression lines from posterior
                x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
                for i in range(0, len(samples), 100):
                    beta = samples[i, :-1]
                    fig1.add_trace(go.Scatter(
                        x=x_range,
                        y=beta[0] * x_range + beta[1],
                        mode='lines',
                        line=dict(color='red', width=1, opacity=0.1),
                        showlegend=False
                    ))
                
                fig1.update_layout(
                    title="Regression Lines from Posterior Samples",
                    xaxis_title="X₁",
                    yaxis_title="y"
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Posterior distribution
                fig2 = go.Figure()
                
                fig2.add_trace(go.Histogram2d(
                    x=samples[:, 0],
                    y=samples[:, 1],
                    nbinsx=50,
                    nbinsy=50,
                    colorscale='Viridis'
                ))
                
                fig2.update_layout(
                    title="Posterior Distribution of Parameters",
                    xaxis_title="β₁",
                    yaxis_title="β₂"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Display results
            st.success(f"Acceptance Rate: {acceptance_rate:.2%}")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.metric(
                    "β₁ (True value: 1.5)",
                    f"{samples[:, 0].mean():.3f} ± {samples[:, 0].std():.3f}"
                )
            
            with col2:
                st.metric(
                    "β₂ (True value: -0.8)",
                    f"{samples[:, 1].mean():.3f} ± {samples[:, 1].std():.3f}"
                )
            
            with col3:
                st.metric(
                    "σ (True value: {noise_std})",
                    f"{samples[:, 2].mean():.3f} ± {samples[:, 2].std():.3f}"
                )

# Add app documentation in an expander
with st.expander("📖 Documentation"):
    st.markdown("""
    ### About this App
    
    This interactive application demonstrates two key concepts in Bayesian statistics:
    
    1. **Metropolis-Hastings Sampling**: Visualizes how MCMC can sample from complex probability distributions.
    2. **Bayesian Linear Regression**: Shows how MCMC can be used for probabilistic inference in regression problems.
    
    ### How to Use
    
    #### Metropolis-Hastings Demo:
    - Adjust the number of samples and proposal width to see how they affect the sampling quality
    - The acceptance rate should typically be between 20% and 50%
    
    #### Bayesian Regression Demo:
    - Modify the data generation parameters to see how they affect the inference
    - The regression plot shows multiple lines sampled from the posterior distribution
    - The heatmap shows the joint posterior distribution of the regression coefficients
    
    ### Implementation Details
    
    The code uses:
    - Numpy for numerical computations
    - Plotly for interactive visualizations
    - Streamlit for the web interface
    """)

# Add footer
st.markdown("""
---
Created with ❤️ using Streamlit
""")
