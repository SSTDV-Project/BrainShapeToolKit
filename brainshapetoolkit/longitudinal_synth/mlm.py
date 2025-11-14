import pickle
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from typing import Dict, List, Optional, Generator, Tuple, Any


class MLMLongitudinalSynthesizer:
    """
    Mixed Linear Model based longitudinal brain shape synthesizer.
    Uses PCA-reduced shape coefficients as dependent variables in mixed-effects models.
    """
    
    def __init__(self, pca_synthesizer, **kwargs):
        """
        Args:
            pca_synthesizer: PCAShapeSynthesizer object with trained PCA
            **kwargs: Optional pre-trained model components
                - mlm_results: List of fitted statsmodels MixedLM results
                - pca_components: List of component names (e.g., ['coeff0', 'coeff1', ...])
        """
        self.pca_synthesizer = pca_synthesizer
        self.mlm_results = kwargs.get('mlm_results', None)
        self.pca_components = kwargs.get('pca_components', None)
        
        # Validate PCA synthesizer has been trained
        if self.pca_synthesizer.pca is None:
            raise ValueError("pca_synthesizer must have a trained PCA model")
        
        # Initialize component names if not provided
        if self.pca_components is None and self.pca_synthesizer.pca is not None:
            n_components = self.pca_synthesizer.pca.n_components_
            self.pca_components = [f'coeff{i}' for i in range(n_components)]
    
    def save(self, path):
        """Save the trained MLM synthesizer to disk."""
        state = {
            'pca_synthesizer': self.pca_synthesizer,
            'mlm_results': self.mlm_results,
            'pca_components': self.pca_components,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f, protocol=5)
    
    @classmethod
    def load(cls, path):
        """Load a trained MLM synthesizer from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        pca_synthesizer = state.pop('pca_synthesizer')
        return cls(pca_synthesizer, **state)
    
    def train(self, 
              subjects_generator: Generator,
              metadata: pd.DataFrame,
              formula: str = "months * group + baseage + sex",
              groups_col: str = "subject",
              verbose: bool = True):
        """
        Train mixed-effects models for each PCA component.
        
        Args:
            subjects_generator: Generator yielding surfaces {'pial': (V,F), 'white': (V,F), ...}
                               Must yield same number of items as rows in metadata
            metadata: DataFrame with columns matching formula variables + groups_col
                     Required columns depend on formula, typically:
                     - subject: subject ID (for random effects)
                     - months: time from baseline
                     - group: diagnostic group (e.g., 'AD', 'MCI', 'CN')
                     - baseage: baseline age
                     - sex: biological sex ('M'/'F')
            formula: Right-hand side of regression formula (excluding dependent variable)
            groups_col: Column name for grouping variable (random intercepts)
            verbose: Print model summaries during training
        
        Returns:
            List of fitted MixedLM result objects
        """
        # Get PCA coefficients for all subjects
        deforms = (self.pca_synthesizer._surfaces_to_deforms(surfaces) 
                   for surfaces in subjects_generator)
        spectvecs = (self.pca_synthesizer._deforms_to_spectvec(d) for d in deforms)
        datapoints = np.stack([*spectvecs], axis=0)
        
        # Transform using pre-trained PCA
        subject_latents = self.pca_synthesizer.pca.transform(datapoints)
        
        # Build training dataframe
        n_components = subject_latents.shape[1]
        train_df = metadata.copy()
        
        # Add PCA coefficients
        for i in range(n_components):
            train_df[self.pca_components[i]] = subject_latents[:, i]
        
        # Ensure categorical variables are properly typed
        if 'group' in train_df.columns:
            train_df['group'] = train_df['group'].astype('category')
        if 'sex' in train_df.columns:
            train_df['sex'] = train_df['sex'].astype('category')
        if groups_col in train_df.columns:
            train_df[groups_col] = train_df[groups_col].astype('category')
        
        # Fit mixed-effects model for each component
        self.mlm_results = []
        
        if verbose:
            print(f"--- Training {n_components} Mixed Linear Models ---\n")
        
        for comp in self.pca_components:
            if verbose:
                print(f"Fitting model for: {comp}")
            
            # Build formula: component ~ fixed_effects + (1|subject)
            full_formula = f'{comp} ~ {formula}'
            
            # Fit mixed linear model
            model = smf.mixedlm(full_formula, train_df, groups=train_df[groups_col])
            result = model.fit()
            
            if verbose:
                print(result.summary())
                print("\n")
            
            self.mlm_results.append(result)
        
        if verbose:
            print("--- Training Complete ---")
        
        return self.mlm_results
    
    def predict(self, 
                predict_df: pd.DataFrame,
                subject_specific: bool = False) -> np.ndarray:
        """
        Predict PCA coefficients for new observations.
        
        Args:
            predict_df: DataFrame with predictor variables matching training formula
                       Must include all fixed effects variables
                       For subject_specific=True, must include 'subject' column with
                       IDs that were present during training
            subject_specific: If True, include subject-specific random effects
                            If False, return population-level predictions
        
        Returns:
            Array of shape (n_obs, n_components) with predicted PCA coefficients
        """
        if self.mlm_results is None:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        n_obs = len(predict_df)
        n_components = len(self.pca_components)
        predictions = np.zeros((n_obs, n_components))
        
        for i, result in enumerate(self.mlm_results):
            if subject_specific:
                # Include random effects for known subjects
                predictions[:, i] = result.predict(predict_df)
            else:
                # Population-level predictions only (random effects = 0)
                predictions[:, i] = result.predict(predict_df)
        
        return predictions
    
    def synthesize(self, 
                   predict_df: pd.DataFrame,
                   subject_specific: bool = False) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Synthesize full brain surfaces for new observations.
        
        Args:
            predict_df: DataFrame with predictor variables
            subject_specific: Include subject-specific random effects
        
        Returns:
            List of surface dictionaries, one per row in predict_df
            Each surface dict: {key: (vertices, faces), ...}
        """
        # Get predicted PCA coefficients
        predicted_latents = self.predict(predict_df, subject_specific=subject_specific)
        
        # Synthesize surfaces using PCA synthesizer
        sample_surfaces = self.pca_synthesizer.sample(sample_latents=predicted_latents)
        
        return sample_surfaces
    
    def synthesize_trajectory(self,
                             subject_id: str,
                             baseline_age: float,
                             group: str,
                             sex: str,
                             months: np.ndarray,
                             subject_specific: bool = False) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Synthesize a longitudinal trajectory for a single subject.
        
        Args:
            subject_id: Unique subject identifier
            baseline_age: Age at baseline
            group: Diagnostic group (e.g., 'AD', 'MCI', 'CN')
            sex: Biological sex ('M' or 'F')
            months: Array of timepoints (months from baseline)
            subject_specific: Use subject-specific effects (only for known subjects)
        
        Returns:
            List of surfaces, one per timepoint
        """
        predict_df = pd.DataFrame({
            'subject': subject_id,
            'baseage': baseline_age,
            'group': group,
            'sex': sex,
            'months': months,
        })
        
        return self.synthesize(predict_df, subject_specific=subject_specific)
    
    def get_model_summary(self, component_idx: int = None) -> str:
        """
        Get summary statistics for fitted models.
        
        Args:
            component_idx: If specified, return summary for specific component
                          If None, return summaries for all components
        
        Returns:
            String with model summary/summaries
        """
        if self.mlm_results is None:
            return "No trained models available"
        
        if component_idx is not None:
            return str(self.mlm_results[component_idx].summary())
        
        summaries = []
        for i, result in enumerate(self.mlm_results):
            summaries.append(f"\n{'='*60}")
            summaries.append(f"Component: {self.pca_components[i]}")
            summaries.append(f"{'='*60}")
            summaries.append(str(result.summary()))
        
        return "\n".join(summaries)