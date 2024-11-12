import streamlit as st
from pathlib import Path
import base64
from PIL import Image
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="MCMC Demo - Documentation",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .header-style {
        font-size: 32px;
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .subheader-style {
        font-size: 24px;
        margin-top: 30px;
        margin-bottom: 20px;
        color: #0f52ba;
    }
    .code-box {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='header-style'>📚 MCMC & Bayesian Inference Demo Documentation</div>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Installation", "Usage Guide", "Technical Details", "Examples", "Troubleshooting"]
)

# Function to display code with syntax highlighting
def display_code(code, language="python"):
    st.code(code, language=language)

# Function to create expandable sections
def create_expandable_section(title, content):
    with st.expander(title):
        st.markdown(content)

# Overview Page
if page == "Overview":
    st.markdown("## 🎯 Overview")
    
    # Features section with cards
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎲 Metropolis-Hastings Sampling
        - Interactive sampling demonstration
        - Mixture of Gaussians visualization
        - Real-time acceptance rate monitoring
        - Adjustable sampling parameters
        """)
        
    with col2:
        st.markdown("""
        #### 📊 Bayesian Linear Regression
        - Synthetic data generation
        - Parameter posterior visualization
        - Uncertainty quantification
        - Interactive plot controls
        """)
    
    # Technology stack
    st.markdown("### 🛠️ Technology Stack")
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        - Python 3.8+
        - Streamlit 1.29.0
        """)
    
    with tech_col2:
        st.markdown("""
        - NumPy
        - SciPy
        """)
    
    with tech_col3:
        st.markdown("""
        - Plotly
        - Pandas
        """)

# Installation Page
elif page == "Installation":
    st.markdown("## 🚀 Installation")
    
    # Prerequisites
    st.markdown("### Prerequisites")
    st.markdown("""
    - Python 3.8 or higher
    - pip (Python package installer)
    - Git (optional, for cloning)
    """)
    
    # Step by step installation
    st.markdown("### Installation Steps")
    
    create_expandable_section(
        "1. Clone Repository or Create Project Directory",
        """
        ```bash
        # Option 1: Clone repository
        git clone https://github.com/yourusername/mcmc-demo.git
        cd mcmc-demo

        # Option 2: Create new directory
        mkdir mcmc-demo
        cd mcmc-demo
        ```
        """
    )
    
    create_expandable_section(
        "2. Create Virtual Environment",
        """
        ```bash
        # Windows
        python -m venv venv
        venv\\Scripts\\activate

        # macOS/Linux
        python3 -m venv venv
        source venv/bin/activate
        ```
        """
    )
    
    create_expandable_section(
        "3. Install Dependencies",
        """
        ```bash
        pip install -r requirements.txt
        ```
        
        Contents of requirements.txt:
        ```
        streamlit==1.29.0
        numpy==1.24.3
        plotly==5.18.0
        scipy==1.11.3
        pandas==2.1.3
        ```
        """
    )

# Usage Guide
elif page == "Usage Guide":
    st.markdown("## 💻 Usage Guide")
    
    # Running the app
    st.markdown("### 🚀 Running the Application")
    display_code("streamlit run app.py", "bash")
    
    # Demo sections
    st.markdown("### 🎮 Using the Demos")
    
    tab1, tab2 = st.tabs(["Metropolis-Hastings Demo", "Bayesian Regression Demo"])
    
    with tab1:
        st.markdown("""
        #### Metropolis-Hastings Sampling
        1. Select "Metropolis-Hastings Sampling" from sidebar
        2. Configure parameters:
            - Number of samples (1,000 - 50,000)
            - Proposal width (0.1 - 2.0)
            - Initial value
        3. Click "Run Sampling"
        4. Analyze results in interactive plots
        """)
        
    with tab2:
        st.markdown("""
        #### Bayesian Linear Regression
        1. Select "Bayesian Linear Regression" from sidebar
        2. Set parameters:
            - Data points (50 - 500)
            - Noise level (0.1 - 2.0)
            - MCMC samples (1,000 - 20,000)
            - Random seed
        3. Click "Run Bayesian Regression"
        4. Examine results and visualizations
        """)

# Technical Details
elif page == "Technical Details":
    st.markdown("## 🔧 Technical Implementation")
    
    # Implementation sections
    tab1, tab2 = st.tabs(["MCMC Implementation", "Visualization Details"])
    
    with tab1:
        st.markdown("### Metropolis-Hastings Algorithm")
        display_code("""
class MetropolisHastings:
    def __init__(self, target_distribution, proposal_width=1.0):
        self.target_distribution = target_distribution
        self.proposal_width = proposal_width
    
    def sample(self, n_samples, initial_value):
        samples = np.zeros(n_samples)
        current = initial_value
        accepted = 0
        
        for i in range(n_samples):
            # Generate proposal
            proposal = current + np.random.normal(0, self.proposal_width)
            
            # Calculate acceptance ratio
            log_ratio = (self.target_distribution(proposal) - 
                        self.target_distribution(current))
            
            # Accept or reject
            if np.log(np.random.random()) < log_ratio:
                current = proposal
                accepted += 1
            
            samples[i] = current
            
        return samples, accepted / n_samples
        """)
        
    with tab2:
        st.markdown("### Visualization Implementation")
        st.markdown("""
        The app uses Plotly for interactive visualizations:
        - Real-time updates
        - Interactive zooming and panning
        - Hover information
        - Downloadable plots
        """)
        display_code("""
# Example of Plotly visualization code
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
        """)

# Examples
elif page == "Examples":
    st.markdown("## 📈 Examples")
    
    # Example 1
    create_expandable_section(
        "Example 1: Simple Mixture Model",
        """
        ```python
        # Define target distribution
        def target_log_pdf(x):
            return np.log(0.5 * norm.pdf(x, -2, 1) + 
                         0.5 * norm.pdf(x, 2, 1.5))

        # Create sampler
        mh = MetropolisHastings(target_log_pdf, proposal_width=0.5)
        
        # Generate samples
        samples, acceptance_rate = mh.sample(
            n_samples=10000, 
            initial_value=0.0
        )
        ```
        """
    )
    
    # Example 2
    create_expandable_section(
        "Example 2: Bayesian Regression",
        """
        ```python
        # Generate synthetic data
        X, y, true_beta = generate_synthetic_data(
            n_samples=100, 
            noise_std=0.5
        )

        # Create and fit model
        model = BayesianLinearRegression(X, y)
        samples, acceptance_rate = model.sample_posterior(
            n_samples=5000,
            proposal_width=0.1
        )
        
        # Access results
        beta1_mean = samples[:, 0].mean()
        beta2_mean = samples[:, 1].mean()
        ```
        """
    )

# Troubleshooting
elif page == "Troubleshooting":
    st.markdown("## ❗ Troubleshooting")
    
    # Common issues
    st.markdown("### Common Issues")
    
    issues = {
        "ImportError: No module named 'streamlit'": """
        **Solution:**
        1. Ensure virtual environment is activated
        2. Reinstall requirements:
        ```bash
        pip install -r requirements.txt
        ```
        """,
        
        "Low Acceptance Rate": """
        **Solution:**
        1. Reduce proposal width
        2. Aim for 20-50% acceptance rate
        3. Current proposal width: try reducing by 50%
        """,
        
        "Slow Performance": """
        **Solution:**
        1. Reduce number of samples
        2. Decrease data points
        3. Consider using a more powerful machine
        """
    }
    
    for issue, solution in issues.items():
        create_expandable_section(issue, solution)
    
    # Performance tips
    st.markdown("### 💡 Performance Tips")
    st.markdown("""
    1. Start with smaller sample sizes
    2. Adjust proposal width gradually
    3. Monitor acceptance rate
    4. Use reasonable noise levels
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Created with ❤️ using Streamlit</p>
        <p>© 2024 MCMC Demo</p>
    </div>
""", unsafe_allow_html=True)
