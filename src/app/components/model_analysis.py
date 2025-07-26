from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import streamlit as st
import torch
from matplotlib.figure import Figure

from src.core.consts import GRAPHS_ORDER, is_mamba_arch
from src.core.types import (
    MODEL_ARCH,
    MODEL_ARCH_AND_SIZE,
    FeatureCategory,
    TModelSize,
)
from src.experiments.infrastructure.setup_models import get_tokenizer_and_model
from src.utils.streamlit.helpers.component import StreamlitComponent


class ModelAnalysisComponent(StreamlitComponent):
    """Base component for model architecture analysis."""

    def __init__(self, model_arch_size: MODEL_ARCH_AND_SIZE):
        """Initialize the model analysis component.

        Args:
            model_arch_size: The model architecture and size to analyze
        """
        self.model_arch_size = model_arch_size
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model and tokenizer."""
        with st.spinner(f"Loading {self.model_arch_size.model_name}..."):
            self.tokenizer, self.model = get_tokenizer_and_model(
                self.model_arch_size.arch,
                self.model_arch_size.size,
                # device="cuda" if torch.cuda.is_available() else "cpu",
                device="cpu",
            )

    def _get_mamba_layers(self):
        """Get all mamba layers from the model."""
        if not self.model:
            self.load_model()

        mamba_layers = []

        # Handle different architectures
        if self.model_arch_size.arch == MODEL_ARCH.MAMBA1:
            # For Mamba 1, extract layers from the model
            for layer in self.model.backbone.layers:
                if hasattr(layer, "mixer") and hasattr(layer.mixer, "A_log"):
                    mamba_layers.append(layer.mixer)
        elif self.model_arch_size.arch == MODEL_ARCH.MAMBA2:
            # For Mamba 2, extract layers from the model
            for layer in self.model.backbone.layers:
                if hasattr(layer, "mixer") and hasattr(layer.mixer, "A_log"):
                    mamba_layers.append(layer.mixer)

        return mamba_layers

    def render(self):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


class AMatrixAnalysisComponent(ModelAnalysisComponent):
    """Component for analyzing A matrices in Mamba models."""

    def __init__(self, model_arch_size: MODEL_ARCH_AND_SIZE):
        super().__init__(model_arch_size)

    def _get_decay_matrices(self, layer_idx: Optional[int] = None):
        """Get decay matrices from specified or all layers.

        Args:
            layer_idx: Optional layer index to get decay matrices from

        Returns:
            Dictionary of layer index to decay matrices
        """
        mamba_layers = self._get_mamba_layers()

        decay_matrices = {}
        if layer_idx is not None and 0 <= layer_idx < len(mamba_layers):
            # Get decay matrices for specific layer
            layer = mamba_layers[layer_idx]
            decay_matrices[layer_idx] = torch.exp(-torch.exp(layer.A_log))
            if self.model_arch_size.arch == MODEL_ARCH.MAMBA2:
                decay_matrices[layer_idx] = decay_matrices[layer_idx].unsqueeze(-1)
        else:
            # Get decay matrices for all layers
            for i, layer in enumerate(mamba_layers):
                decay_matrices[i] = torch.exp(-torch.exp(layer.A_log))

        return decay_matrices

    def _calculate_decay_stats(self, decay_matrices: Dict[int, torch.Tensor]):
        """Calculate statistics on decay matrices.

        Args:
            decay_matrices: Dictionary of layer index to decay matrices

        Returns:
            Dictionary of statistics
        """
        stats = {}

        for layer_idx, decay in decay_matrices.items():
            norms = torch.norm(decay, p=1, dim=1)

            # Calculate statistics
            stats[layer_idx] = {
                "min": norms.min().item(),
                "max": norms.max().item(),
                "mean": norms.mean().item(),
                "median": norms.median().item(),
                "std": norms.std().item(),
                "hist": norms.detach().cpu().numpy(),
            }

        return stats

    def plot_decay_histogram(self, layer_idx: int, division_factor: int = 3) -> Figure:
        """Plot histogram of decay norms for a specific layer.

        Args:
            layer_idx: Layer index
            division_factor: Division factor for highlighting the split

        Returns:
            Matplotlib figure
        """
        decay_matrices = self._get_decay_matrices(layer_idx)
        decay = decay_matrices[layer_idx]
        norms = torch.norm(decay, p=1, dim=1)

        # Sort norms
        sorted_norms, sorted_indices = torch.sort(norms)
        n_ssms = len(sorted_norms)
        split_idx = n_ssms // division_factor

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        counts, bins, _ = ax.hist(
            norms.detach().cpu().numpy(),
            bins=30,
            alpha=0.7,
            color="blue",
            label="All features",
        )

        # Add vertical line for the split
        split_value = sorted_norms[split_idx].item()
        ax.axvline(
            x=split_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Split at 1/{division_factor} ({split_idx}/{n_ssms} features)",
        )

        # Add annotations
        ax.set_title(f"Layer {layer_idx}: Decay Matrix Norm Distribution")
        ax.set_xlabel("L1 Norm of Decay Matrix")
        ax.set_ylabel("Count")
        ax.legend()

        return fig

    def plot_all_layers_boxplot(self) -> Figure:
        """Plot boxplot of decay norms across all layers.

        Returns:
            Matplotlib figure
        """
        decay_matrices = self._get_decay_matrices()

        # Prepare data for boxplot
        boxplot_data = []
        for layer_idx in sorted(decay_matrices.keys()):
            norms = torch.norm(decay_matrices[layer_idx], p=1, dim=1)
            boxplot_data.append(norms.detach().cpu().numpy())

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.boxplot(boxplot_data)

        # Add annotations
        ax.set_title("Decay Matrix Norm Distribution Across Layers")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("L1 Norm of Decay Matrix")
        ax.set_xticklabels([str(i) for i in sorted(decay_matrices.keys())])

        return fig

    def analyze_division_factors(self, layer_idx: int, factors: List[int] = [2, 3, 4, 5, 10]):
        """Analyze different division factors for feature selection.

        Args:
            layer_idx: Layer index
            factors: List of division factors to analyze
        """
        decay_matrices = self._get_decay_matrices(layer_idx)
        decay = decay_matrices[layer_idx]
        norms = torch.norm(decay, p=1, dim=1)

        # Sort norms
        sorted_norms, sorted_indices = torch.sort(norms)
        n_ssms = len(sorted_norms)

        # Create figure with subplots for each factor
        fig, axs = plt.subplots(len(factors), 1, figsize=(10, 4 * len(factors)))

        for i, factor in enumerate(factors):
            split_idx = n_ssms // factor
            ax = axs[i] if len(factors) > 1 else axs

            # Plot histogram
            ax.hist(norms.detach().cpu().numpy(), bins=30, alpha=0.7)
            split_value = sorted_norms[split_idx].item()

            # Add vertical line
            ax.axvline(x=split_value, color="red", linestyle="--", linewidth=2)

            # Add annotations
            ax.set_title(f"Division Factor: 1/{factor} (Features: {split_idx}/{n_ssms})")
            ax.set_xlabel("L1 Norm")
            ax.set_ylabel("Count")

        fig.tight_layout()
        return fig

    def render(self):
        """Render the component."""
        st.write(f"## A Matrix Analysis for {self.model_arch_size.model_name}")

        # Load model if not already loaded
        if not self.model:
            self.load_model()

        mamba_layers = self._get_mamba_layers()
        num_layers = len(mamba_layers)

        if num_layers == 0:
            st.error(f"No Mamba layers found in {self.model_arch_size.model_name}")
            return

        st.write(f"Found {num_layers} Mamba layers")

        # Layer selection
        layer_idx = st.slider("Select Layer", 0, num_layers - 1, 0)

        # Division factor selection
        division_factor = st.slider(
            "Division Factor (1/n of features)",
            min_value=2,
            max_value=10,
            value=3,
            help="Features are split into fast/slow decay based on this division factor",
        )

        # Display basic histogram
        st.write(f"### Layer {layer_idx} Decay Matrix Analysis")
        histogram_fig = self.plot_decay_histogram(layer_idx, division_factor)
        st.pyplot(histogram_fig)

        # Display division factor analysis
        with st.expander("Analyze Different Division Factors"):
            factors = [2, 3, 4, 5, 10]
            st.write("### Impact of Different Division Factors")
            division_analysis_fig = self.analyze_division_factors(layer_idx, factors)
            st.pyplot(division_analysis_fig)

        # Display across-layer analysis
        with st.expander("Analyze Across All Layers"):
            st.write("### Decay Matrix Distribution Across All Layers")
            boxplot_fig = self.plot_all_layers_boxplot()
            st.pyplot(boxplot_fig)


class FeatureDynamicsComponent(ModelAnalysisComponent):
    """Component for analyzing feature dynamics in Mamba models."""

    def __init__(self, model_arch_size: MODEL_ARCH_AND_SIZE):
        super().__init__(model_arch_size)

    def _analyze_feature_dynamics(
        self,
        layer_idx: int,
        input_text: str = "The quick brown fox jumps over the lazy dog",
    ):
        """Analyze feature dynamics for a specific layer."""
        if not self.model:
            self.load_model()

        mamba_layers = self._get_mamba_layers()

        if layer_idx >= len(mamba_layers):
            return None

        layer = mamba_layers[layer_idx]

        # Tokenize input and get model outputs
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # Store original state
        original_mode = self.model.training
        self.model.eval()

        # Add hooks to capture intermediate outputs
        hidden_states = {}

        def hook_fn(module, input, output):
            hidden_states["output"] = output

        handle = layer.register_forward_hook(hook_fn)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Remove hook
        handle.remove()

        # Restore original mode
        self.model.train(original_mode)

        # Get A matrices and hidden states
        A_log = layer.A_log
        decay_matrices = torch.exp(-torch.exp(A_log))

        return {
            "decay_matrices": decay_matrices.detach().cpu(),
            "hidden_states": (hidden_states["output"].detach().cpu() if "output" in hidden_states else None),
            "input_tokens": inputs["input_ids"][0].detach().cpu(),
        }

    def visualize_feature_dynamics(self, layer_idx: int, input_text: str, feature_category: FeatureCategory):
        """Visualize feature dynamics for specific category."""
        results = self._analyze_feature_dynamics(layer_idx, input_text)

        if not results or results["hidden_states"] is None:
            return None

        # Get decay matrices
        decay_matrices = results["decay_matrices"]
        n_ssms = decay_matrices.shape[0]

        # Get norms
        norms = torch.norm(decay_matrices, p=1, dim=1)

        # Sort features based on category
        sorted_indices = torch.argsort(norms, descending=(feature_category == FeatureCategory.SLOW_DECAY))

        # Get top 1/3 features
        selected_indices = sorted_indices[: n_ssms // 3]

        # Create visualization
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))

        # Plot decay distribution
        axs[0].hist(norms.numpy(), bins=30, alpha=0.7)
        axs[0].set_title(f"Decay Distribution - Layer {layer_idx}")
        axs[0].set_xlabel("L1 Norm")
        axs[0].set_ylabel("Count")

        # Highlight selected features
        selected_norms = norms[selected_indices]
        axs[0].hist(selected_norms.numpy(), bins=30, alpha=0.7, color="red")

        # Add legend
        if feature_category == FeatureCategory.SLOW_DECAY:
            axs[0].legend(["All Features", "Slow Decay Features (Top 1/3)"])
        else:
            axs[0].legend(["All Features", "Fast Decay Features (Bottom 1/3)"])

        # Plot hidden states activations (average across sequence)
        hidden_states = results["hidden_states"]

        # Get mean activation for each feature across sequence
        mean_activations = torch.mean(hidden_states, dim=1)  # Average across sequence

        # Plot selected feature activations
        axs[1].bar(
            range(len(selected_indices)),
            mean_activations[0, selected_indices].numpy(),
            alpha=0.7,
        )
        axs[1].set_title(
            "Feature Activations for"
            f" {'Slow' if feature_category == FeatureCategory.SLOW_DECAY else 'Fast'}"
            "Decay Features"
        )
        axs[1].set_xlabel("Feature Index (sorted by norm)")
        axs[1].set_ylabel("Mean Activation")

        fig.tight_layout()
        return fig

    def render(self):
        """Render the component."""
        st.write(f"## Feature Dynamics Analysis for {self.model_arch_size.model_name}")

        # Load model if not already loaded
        if not self.model:
            self.load_model()

        mamba_layers = self._get_mamba_layers()
        num_layers = len(mamba_layers)

        if num_layers == 0:
            st.error(f"No Mamba layers found in {self.model_arch_size.model_name}")
            return

        st.write(f"Found {num_layers} Mamba layers")

        # Layer selection
        layer_idx = st.slider("Select Layer", 0, num_layers - 1, 0, key="feature_dynamics_layer")

        # Input text
        input_text = st.text_area(
            "Input Text",
            "The quick brown fox jumps over the lazy dog",
            key="feature_dynamics_input",
        )

        # Feature category selection
        category_options = {
            "Fast Decay": FeatureCategory.FAST_DECAY,
            "Slow Decay": FeatureCategory.SLOW_DECAY,
        }

        selected_category = st.radio(
            "Feature Category",
            list(category_options.keys()),
            index=1,  # Default to Slow Decay
            key="feature_dynamics_category",
        )

        feature_category = category_options[selected_category]

        # Analyze button
        if st.button("Analyze Feature Dynamics"):
            with st.spinner("Analyzing feature dynamics..."):
                fig = self.visualize_feature_dynamics(layer_idx, input_text, feature_category)
                if fig:
                    st.pyplot(fig)
                else:
                    st.error("Failed to analyze feature dynamics")


def select_mamba_model() -> MODEL_ARCH_AND_SIZE:
    """Select a Mamba model from available models.

    Returns:
        Selected model architecture and size
    """
    # Filter for Mamba models
    mamba_models = [model for model in GRAPHS_ORDER.keys() if is_mamba_arch(model.arch)]

    # Default to Mamba1 130M if available
    default_model = next(
        (m for m in mamba_models if m.arch == MODEL_ARCH.MAMBA1 and m.size == TModelSize("130M")),
        mamba_models[0] if mamba_models else None,
    )

    model_options = {f"{model.arch.value}-{model.size}": model for model in mamba_models}

    # Find default index
    default_index = 0
    if default_model:
        default_name = f"{default_model.arch.value}-{default_model.size}"
        default_index = list(model_options.keys()).index(default_name) if default_name in model_options else 0

    # Model selection
    selected_name = st.selectbox("Select Mamba Model", options=list(model_options.keys()), index=default_index)

    return model_options[selected_name]
