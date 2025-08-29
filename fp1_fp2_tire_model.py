import fastf1
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

def load_practice_sessions(year, gp_name):
    """
    Load FP1 and FP2 sessions and combine their data for tire modeling
    
    Args:
        year: Race year (e.g., 2024)
        gp_name: Grand Prix name (e.g., 'Dutch Grand Prix')
        
    Returns:
        combined_laps: Combined lap data from both sessions
        session_info: Information about data availability
    """
    print(f"Loading practice sessions for {year} {gp_name}...")
    
    combined_laps = []
    session_info = {'fp1_available': False, 'fp2_available': False, 'total_laps': 0}
    
    # Try to load FP1
    try:
        fp1 = fastf1.get_session(year, gp_name, 'FP1')
        fp1.load()
        
        fp1_laps = fp1.laps.copy()
        fp1_laps['Session'] = 'FP1'
        fp1_laps['SessionType'] = 'Practice'
        
        # Filter for clean laps with tire data
        fp1_clean = fp1_laps[
            (fp1_laps['LapTime'].notna()) &
            (fp1_laps['Compound'].notna()) &
            (~fp1_laps['PitOutTime'].notna()) &  # Not pit out lap
            (~fp1_laps['PitInTime'].notna()) &   # Not pit in lap
            (fp1_laps['TrackStatus'] == '1')     # Green flag conditions
        ].copy()
        
        if len(fp1_clean) > 0:
            combined_laps.append(fp1_clean)
            session_info['fp1_available'] = True
            session_info['fp1_laps'] = len(fp1_clean)
            print(f"   FP1: {len(fp1_clean)} clean laps loaded")
        else:
            print(f"   FP1: No clean laps available")
            
    except Exception as e:
        print(f"  FP1: Could not load - {e}")
    
    # Try to load FP2
    try:
        fp2 = fastf1.get_session(year, gp_name, 'FP2')
        fp2.load()
        
        fp2_laps = fp2.laps.copy()
        fp2_laps['Session'] = 'FP2'
        fp2_laps['SessionType'] = 'Practice'
        
        # Filter for clean laps with tire data
        fp2_clean = fp2_laps[
            (fp2_laps['LapTime'].notna()) &
            (fp2_laps['Compound'].notna()) &
            (~fp2_laps['PitOutTime'].notna()) &  # Not pit out lap
            (~fp2_laps['PitInTime'].notna()) &   # Not pit in lap
            (fp2_laps['TrackStatus'] == '1')     # Green flag conditions
        ].copy()
        
        if len(fp2_clean) > 0:
            combined_laps.append(fp2_clean)
            session_info['fp2_available'] = True
            session_info['fp2_laps'] = len(fp2_clean)
            print(f"   FP2: {len(fp2_clean)} clean laps loaded")
        else:
            print(f"   FP2: No clean laps available")
            
    except Exception as e:
        print(f"   FP2: Could not load - {e}")
    
    # Combine all available practice data
    if combined_laps:
        combined_data = pd.concat(combined_laps, ignore_index=True)
        
        # Add stint lap calculation for practice sessions
        # Group by driver, session, and stint to calculate stint lap
        combined_data['StintLap'] = combined_data.groupby(['Driver', 'Session', 'Stint']).cumcount() + 1
        combined_data['LapTime_s'] = combined_data['LapTime'].dt.total_seconds()
        
        session_info['total_laps'] = len(combined_data)
        session_info['compounds_available'] = list(combined_data['Compound'].unique())
        
        print(f"   Combined: {len(combined_data)} total clean practice laps")
        print(f"   Compounds found: {', '.join(session_info['compounds_available'])}")
        
        return combined_data, session_info
    else:
        print(f"   No practice data available for tire modeling")
        return pd.DataFrame(), session_info

def build_tire_model_from_practice(compound_data, compound_name, base_pace=71.0):
    """
    Build tire degradation model from practice session data
    
    Args:
        compound_data: DataFrame with practice lap data for specific compound
        compound_name: Name of the tire compound
        base_pace: Expected base lap time for the track
        
    Returns:
        mcmc: Fitted MCMC model, or None if insufficient data
    """
    if len(compound_data) < 10:  # Minimum laps needed for reliable modeling
        print(f"     {compound_name}: Only {len(compound_data)} laps - insufficient for modeling")
        return None
    
    # Filter out obvious outliers (beyond 3 standard deviations)
    lap_times = compound_data['LapTime_s']
    mean_time = lap_times.mean()
    std_time = lap_times.std()
    
    # Remove extreme outliers
    clean_data = compound_data[
        (lap_times >= mean_time - 3 * std_time) &
        (lap_times <= mean_time + 3 * std_time)
    ].copy()
    
    if len(clean_data) < 8:
        print(f"     {compound_name}: Only {len(clean_data)} clean laps after outlier removal")
        return None
    
    # Prepare data for modeling
    x = clean_data["StintLap"].values
    y = clean_data["LapTime_s"].values
    
    print(f"     {compound_name}: Modeling {len(clean_data)} laps (stint laps 1-{max(x)})")
    
    def tire_model(x, y=None):
        """Bayesian tire degradation model"""
        # Base pace prior (centered on track-specific base pace)
        alpha = numpyro.sample("alpha", dist.Normal(base_pace, 2.0))
        
        # Degradation rate prior (compound-specific expectations)
        if compound_name == 'SOFT':
            beta_prior = dist.Normal(0.15, 0.05)  # Softs degrade faster
        elif compound_name == 'MEDIUM':
            beta_prior = dist.Normal(0.08, 0.03)  # Medium degradation
        elif compound_name == 'HARD':
            beta_prior = dist.Normal(0.04, 0.02)  # Hards degrade slower
        else:  # INTERMEDIATE or other
            beta_prior = dist.Normal(0.06, 0.03)  # Conservative estimate
        
        beta = numpyro.sample("beta", beta_prior)
        
        # Observation noise
        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
        
        # Linear degradation model
        mu = alpha + beta * x
        
        # Likelihood
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
    
    # Run MCMC
    try:
        kernel = NUTS(tire_model)
        mcmc = MCMC(kernel, num_warmup=800, num_samples=1200)
        mcmc.run(random.PRNGKey(42), x, y)
        
        # Extract posterior statistics
        samples = mcmc.get_samples()
        alpha_mean = np.mean(samples['alpha'])
        beta_mean = np.mean(samples['beta'])
        
        print(f"     {compound_name}: Base={alpha_mean:.2f}s, Deg={beta_mean:.4f}s/lap")
        
        return mcmc
        
    except Exception as e:
        print(f"     {compound_name}: Model fitting failed - {e}")
        return None

def build_compound_models_from_practice(year, gp_name, base_pace=71.0):
    """
    Build tire compound models using FP1 and FP2 data
    
    Args:
        year: Year to analyze
        gp_name: Grand Prix name
        base_pace: Expected base lap time for the track
        
    Returns:
        compound_models: Dictionary of fitted models by compound
        model_info: Information about model quality and data sources
    """
    print(f"\n{'='*60}")
    print(f"BUILDING TIRE MODELS FROM PRACTICE DATA")
    print(f"{'='*60}")
    
    # Load practice session data
    practice_data, session_info = load_practice_sessions(year, gp_name)
    
    if practice_data.empty:
        print(" No practice data available - cannot build tire models")
        return {}, {'error': 'No practice data available'}
    
    compound_models = {}
    model_info = {
        'session_info': session_info,
        'compounds_modeled': [],
        'compounds_failed': [],
        'total_practice_laps': len(practice_data),
        'model_quality': {}
    }
    
    # Group data by compound
    compounds = practice_data['Compound'].unique()
    print(f"\n Building models for compounds: {', '.join(compounds)}")
    
    for compound in compounds:
        print(f"\n   Processing {compound} compound...")
        
        compound_data = practice_data[practice_data['Compound'] == compound].copy()
        
        # Show data distribution across sessions
        fp1_laps = len(compound_data[compound_data['Session'] == 'FP1'])
        fp2_laps = len(compound_data[compound_data['Session'] == 'FP2'])
        
        print(f"     Data: FP1={fp1_laps} laps, FP2={fp2_laps} laps, Total={len(compound_data)}")
        
        # Build model
        mcmc_model = build_tire_model_from_practice(compound_data, compound, base_pace)
        
        if mcmc_model is not None:
            compound_models[compound] = mcmc_model
            model_info['compounds_modeled'].append(compound)
            
            # Calculate model quality metrics
            samples = mcmc_model.get_samples()
            alpha_std = np.std(samples['alpha'])
            beta_std = np.std(samples['beta'])
            
            # Quality assessment based on parameter uncertainty
            if alpha_std < 0.5 and beta_std < 0.01:
                quality = 'High'
            elif alpha_std < 1.0 and beta_std < 0.02:
                quality = 'Good'
            else:
                quality = 'Moderate'
            
            model_info['model_quality'][compound] = {
                'quality': quality,
                'alpha_uncertainty': alpha_std,
                'beta_uncertainty': beta_std,
                'data_points': len(compound_data)
            }
            
        else:
            model_info['compounds_failed'].append(compound)
    
    # Summary
    print(f"\n{'='*40}")
    print(f"TIRE MODEL BUILDING SUMMARY")
    print(f"{'='*40}")
    
    print(f" Successfully modeled: {', '.join(model_info['compounds_modeled'])}")
    if model_info['compounds_failed']:
        print(f" Failed to model: {', '.join(model_info['compounds_failed'])}")
    
    print(f"\n Model Quality Assessment:")
    for compound, quality_info in model_info['model_quality'].items():
        quality = quality_info['quality']
        data_points = quality_info['data_points']
        print(f"  {compound}: {quality} quality ({data_points} data points)")
    
    # Data source summary
    print(f"\n Data Sources:")
    if session_info['fp1_available']:
        print(f"  FP1: {session_info.get('fp1_laps', 0)} laps")
    if session_info['fp2_available']:
        print(f"  FP2: {session_info.get('fp2_laps', 0)} laps")
    print(f"  Total: {session_info['total_laps']} practice laps")
    
    return compound_models, model_info

# Updated tire performance function that uses practice-based models
def get_tire_performance_from_practice_models(compound, lap_in_stint, compound_models, 
                                            base_pace=71.0, weather='dry', track_evolution=0,
                                            extracted_params=None):
    """
    Calculate lap time using practice-based tire models
    
    Args:
        compound: Tire compound name
        lap_in_stint: Current lap in the stint
        compound_models: Dictionary of fitted MCMC models
        base_pace: Fallback base pace if model not available
        weather: Weather conditions ('dry' or 'wet')
        track_evolution: Track evolution factor (negative = getting faster)
        extracted_params: Dictionary containing extracted parameters (optional)
        
    Returns:
        predicted_lap_time: Predicted lap time in seconds
    """
    
    if weather == 'wet':
        # For wet conditions, use simplified model (practice usually dry)
        if compound == 'INTERMEDIATE':
            base_time = base_pace + 9.0  # Wet conditions penalty
            degradation = 0.03 * lap_in_stint
        else:
            # Wrong tire for conditions - heavy penalty
            base_time = base_pace + 15.0
            degradation = 0.1 * lap_in_stint
            
        return base_time + degradation + track_evolution
    
    # Dry conditions - use practice models if available
    if compound in compound_models:
        try:
            # Use the fitted model to predict lap time
            samples = compound_models[compound].get_samples()
            
            # Use median values from posterior
            alpha_median = np.median(samples['alpha'])
            beta_median = np.median(samples['beta'])
            
            base_time = alpha_median
            degradation = beta_median * lap_in_stint
            
        except Exception as e:
            print(f"Warning: Could not use practice model for {compound}: {e}")
            # Fallback to extracted parameters or defaults
            base_time, degradation = _get_fallback_tire_performance(
                compound, lap_in_stint, base_pace, extracted_params
            )
    else:
        # No practice model available - use extracted parameters or fallback
        base_time, degradation = _get_fallback_tire_performance(
            compound, lap_in_stint, base_pace, extracted_params
        )
    
    # Apply track evolution
    base_time += track_evolution
    
    # Non-linear degradation for very long stints
    if lap_in_stint > 25:
        degradation += 0.04 * (lap_in_stint - 25) ** 1.3
    
    return base_time + degradation

def _get_fallback_tire_performance(compound, lap_in_stint, base_pace, extracted_params=None):
    """Helper function for fallback tire performance calculation"""
    
    # Try to use extracted parameters if provided
    if extracted_params and 'TIRE_PERFORMANCE' in extracted_params:
        tire_perf = extracted_params['TIRE_PERFORMANCE']
        if compound in tire_perf:
            compound_offset = tire_perf[compound].get('offset', 0.0)
            deg_rate = tire_perf[compound].get('degradation_rate', 0.08)
            base_time = base_pace + compound_offset
            degradation = deg_rate * lap_in_stint
            return base_time, degradation
    
    # Ultimate fallback to hardcoded values
    offsets = {'SOFT': 0.0, 'MEDIUM': 0.35, 'HARD': 0.65}
    deg_rates = {'SOFT': 0.15, 'MEDIUM': 0.08, 'HARD': 0.04}
    base_time = base_pace + offsets.get(compound, 0.0)
    degradation = deg_rates.get(compound, 0.08) * lap_in_stint
    
    return base_time, degradation

# Example usage function
def main():
    """Example of how to use the updated tire modeling system"""
    
    # Build models from practice sessions
    year = 2025
    gp_name = 'Dutch Grand Prix'
    base_pace = 71.0  # Zandvoort lap time
    
    compound_models, model_info = build_compound_models_from_practice(year, gp_name, base_pace)
    
    if compound_models:
        print(f"\n{'='*50}")
        print("TESTING PRACTICE-BASED TIRE MODELS")
        print('='*50)
        
        # Test predictions for different compounds and stint laps
        test_compounds = ['SOFT', 'MEDIUM', 'HARD']
        test_stint_laps = [1, 10, 20, 30]
        
        for compound in test_compounds:
            if compound in compound_models:
                print(f"\n{compound} Tire Predictions:")
                for stint_lap in test_stint_laps:
                    predicted_time = get_tire_performance_from_practice_models(
                        compound, stint_lap, compound_models, base_pace
                    )
                    print(f"  Stint lap {stint_lap}: {predicted_time:.3f}s")
            else:
                print(f"\n{compound}: No practice model available")
    
    else:
        print(" No tire models could be built from practice data")

if __name__ == "__main__":
    main()