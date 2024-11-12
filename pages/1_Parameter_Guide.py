import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# Page config
st.set_page_config(
    page_title="MCMC Demo - Parameter Guide",
    page_icon="🎛️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .param-title {
        font-size: 24px;
        color: #0f52ba;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .effect-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .tip-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎛️ Parameter Guide & Effects")

# Navigation
section = st.sidebar.radio(
    "Section",
    ["Metropolis-Hastings Parameters", 
     "Bayesian Regression Parameters",
     "Understanding Posterior Distributions"]
)

def plot_proposal_distributions(mu, sigmas):
    """Create visualization of proposal distributions"""
    x = np.linspace(mu-5, mu+5, 200)
    fig = go.Figure()
    
    for sigma in sigmas:
        y = norm.pdf(x, mu, sigma)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            name=f'σ={sigma}',
            mode='lines'
        ))
    
    fig.update_layout(
        title='Proposal Distributions with Different Widths',
        xaxis_title='x',
        yaxis_title='Density',
        showlegend=True
    )
    return fig

if section == "Metropolis-Hastings Parameters":
    st.markdown("## Metropolis-Hastings Algorithm Parameters")
    
    st.markdown("""
    The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo (MCMC) method 
    that allows us to sample from complex probability distributions. Understanding 
    its parameters is crucial for effective sampling.
    """)
    
    # Number of Samples
    st.markdown("<div class='param-title'>1. Number of Samples</div>", 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("""
        **What it is:**  
        The total number of points to sample from the target distribution.
        
        **Effects of changing this parameter:**
        - **Too Few** (<1,000): 
            - Poor representation of the target distribution
            - High variance in estimates
            - Unstable convergence
        - **Sweet Spot** (10,000 - 50,000):
            - Good balance of accuracy and computation time
            - Stable estimates
            - Reliable convergence diagnostics
        - **Too Many** (>100,000):
            - Diminishing returns in accuracy
            - Increased computation time
            - Memory usage concerns
        """)
        
    with col2:
        st.markdown("<div class='effect-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Recommended Range:**  
        - Start with 10,000
        - Adjust based on:
            - Complexity of target
            - Required precision
            - Computational resources
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Proposal Distribution Width
    st.markdown("<div class='param-title'>2. Proposal Distribution Width</div>",
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("""
        **What it is:**  
        The standard deviation of the normal distribution used to propose new samples.
        Controls how far the algorithm can "jump" in each step.
        
        **Effects of changing this parameter:**
        - **Too Small** (<0.1):
            - High acceptance rate but slow exploration
            - Gets stuck in local regions
            - Poor mixing of the chain
        - **Sweet Spot** (0.1 - 1.0):
            - Good balance of exploration and acceptance
            - Efficient mixing
            - Acceptance rate around 20-50%
        - **Too Large** (>2.0):
            - Low acceptance rate
            - Wasteful computation
            - Poor convergence
        """)
        
    with col2:
        st.markdown("<div class='effect-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Recommended Strategy:**
        1. Start with 0.5
        2. Monitor acceptance rate
        3. Adjust to achieve 20-50% acceptance
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualization of proposal distributions
    st.plotly_chart(plot_proposal_distributions(0, [0.5, 1.0, 2.0]))
    
    # Initial Value
    st.markdown("<div class='param-title'>3. Initial Value</div>",
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("""
        **What it is:**  
        The starting point of the Markov chain. The first sample in the sequence.
        
        **Effects of choosing this parameter:**
        - **Near Mode**:
            - Faster convergence
            - Shorter burn-in period
            - More efficient sampling
        - **Far from Mode**:
            - Longer convergence time
            - Needs longer burn-in
            - Good for testing robustness
        - **Multiple Chains**:
            - Start chains at different points
            - Check convergence to same distribution
            - Verify mixing
        """)
        
    with col2:
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Important Note:**  
        The choice of initial value becomes less important as the number of samples increases, due to the ergodic property of MCMC.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

elif section == "Bayesian Regression Parameters":
    st.markdown("## Bayesian Linear Regression Parameters")
    
    st.markdown("""
    Bayesian linear regression extends classical regression by providing full 
    probability distributions for the parameters. Understanding these parameters 
    helps in getting reliable posterior estimates.
    """)
    
    # Number of Data Points
    st.markdown("<div class='param-title'>1. Number of Data Points</div>",
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("""
        **What it is:**  
        The size of the synthetic dataset used for demonstration.
        
        **Effects of changing this parameter:**
        - **Small Dataset** (<50 points):
            - Higher uncertainty in parameter estimates
            - Broader posterior distributions
            - Prior has stronger influence
        - **Medium Dataset** (50-200 points):
            - Good balance of uncertainty and precision
            - Clear demonstration of Bayesian updating
            - Reasonable computation time
        - **Large Dataset** (>200 points):
            - More precise parameter estimates
            - Narrower posterior distributions
            - Prior becomes less influential
        """)
        
    with col2:
        st.markdown("<div class='tip-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Best Practice:**
        - Start with 100 points
        - Increase if uncertainty is too high
        - Decrease to demonstrate prior influence
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Noise Level
    st.markdown("<div class='param-title'>2. Noise Level</div>",
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("""
        **What it is:**  
        The standard deviation of the random noise added to the true relationship.
        Controls how scattered the data points are around the true line.
        
        **Effects of changing this parameter:**
        - **Low Noise** (0.1-0.3):
            - Clear linear pattern
            - Tight posterior distributions
            - Quick convergence
        - **Medium Noise** (0.3-0.8):
            - Realistic data scatter
            - Balanced uncertainty
            - Good for demonstration
        - **High Noise** (>0.8):
            - Difficult to discern pattern
            - Wide posterior distributions
            - Requires more samples
        """)
        
    with col2:
        st.markdown("<div class='effect-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Recommended Values:**
        - Start with 0.5
        - Adjust based on:
            - Desired clarity of pattern
            - Teaching objectives
            - Realism needs
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # MCMC Samples
    st.markdown("<div class='param-title'>3. MCMC Samples</div>",
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("""
        **What it is:**  
        The number of posterior samples to generate using MCMC.
        
        **Effects of changing this parameter:**
        - **Few Samples** (<2,000):
            - Rough posterior approximation
            - Quick computation
            - May miss important regions
        - **Moderate Samples** (2,000-10,000):
            - Good posterior representation
            - Reasonable computation time
            - Stable parameter estimates
        - **Many Samples** (>10,000):
            - Very accurate posteriors
            - Longer computation time
            - Diminishing returns
        """)
        
    with col2:
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Important:**  
        More samples ≠ Better model
        It only means better approximation of the posterior distribution.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

elif section == "Understanding Posterior Distributions":
    st.markdown("## Understanding Posterior Distributions")
    
    st.markdown("""
    The posterior distribution is a fundamental concept in Bayesian statistics. 
    It represents our updated beliefs about parameters after observing data.
    """)
    
    # What is a Posterior Distribution?
    st.markdown("<div class='param-title'>What is a Posterior Distribution?</div>",
                unsafe_allow_html=True)
    
    st.markdown("""
    The posterior distribution combines:
    1. **Prior Distribution**: Our initial beliefs about parameters
    2. **Likelihood**: How well the parameters explain the data
    3. **Bayes' Theorem**: P(parameters|data) ∝ P(data|parameters) × P(parameters)
    """)
    
    # Components and Interpretation
    st.markdown("<div class='param-title'>Components and Interpretation</div>",
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='effect-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Key Components:**
        1. **Location**: Where the bulk of probability lies
            - Mean: Average value
            - Median: Middle value
            - Mode: Most likely value
        
        2. **Spread**: How uncertain we are
            - Standard deviation
            - Credible intervals
            - Probability regions
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='effect-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Interpretation:**
        - Wider distribution = More uncertainty
        - Narrower distribution = More certainty
        - Multiple peaks = Multiple plausible values
        - Skewness = Asymmetric uncertainty
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # How Parameters Affect the Posterior
    st.markdown("<div class='param-title'>How Parameters Affect the Posterior</div>",
                unsafe_allow_html=True)
    
    st.markdown("""
    1. **Data Size Effects:**
        - More data → Narrower posterior
        - Less data → Wider posterior
        - Prior influence decreases with more data
    
    2. **Noise Level Effects:**
        - Higher noise → Wider posterior
        - Lower noise → Narrower posterior
        - Affects uncertainty in parameter estimates
    
    3. **Prior Effects:**
        - Strong prior → Resistant to data
        - Weak prior → Easily influenced by data
        - Prior impact decreases with more data
    """)
    
    # Common Patterns in Posteriors
    st.markdown("<div class='param-title'>Common Patterns in Posteriors</div>",
                unsafe_allow_html=True)
    
    st.markdown("""
    1. **Normal-like Posterior:**
        - Common with large datasets
        - Symmetric uncertainty
        - Easy to summarize
    
    2. **Skewed Posterior:**
        - Common with constrained parameters
        - Asymmetric uncertainty
        - Median ≠ Mean
    
    3. **Multimodal Posterior:**
        - Multiple plausible values
        - Complex parameter space
        - Requires careful interpretation
    """)
    
    # Tips for Interpretation
    st.markdown("<div class='tip-box'>", unsafe_allow_html=True)
    st.markdown("""
    **Tips for Interpretation:**
    1. Always plot the full distribution, not just summary statistics
    2. Consider multiple chains to verify convergence
    3. Use appropriate credible intervals (not just 95%)
    4. Remember that the posterior is a probability distribution
    5. Consider the practical significance of the uncertainty
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Need more help? Check the main documentation or contact support.</p>
    </div>
""", unsafe_allow_html=True)