import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

class F1ParameterExtractor:
    def __init__(self):
        self.cache_dir = './f1_cache'
        fastf1.Cache.enable_cache(self.cache_dir)
        
    def extract_targeted_parameters(self, track_name='Dutch Grand Prix', years=[2022, 2023, 2024]):
        """Extract specific simulation parameters from historical data"""
        print(f"\nExtracting targeted parameters for {track_name} from years {years}")
        print("="*60)
        
        parameters = {}
        
        # Extract only the requested parameter sets
        parameters['position_penalties'] = self.extract_position_penalties(track_name, years)
        parameters['tire_performance'] = self.extract_tire_performance_soft_baseline(track_name, years)
        parameters['driver_error_rates'] = self.extract_driver_error_rates(track_name, years)
        parameters['drs_effectiveness'] = self.extract_drs_effectiveness(track_name, years)
        
        # Generate parameter file
        self.generate_parameter_file(parameters, track_name)
        
        return parameters
    
    def extract_position_penalties(self, track_name, years):
        """Extract time penalties by starting position from qualifying vs race pace comparison"""
        print(f"\n1. Extracting position penalties...")
        
        position_effects = []
        
        for year in years:
            try:
                # Get qualifying and race data
                quali = fastf1.get_session(year, track_name, 'Q')
                race = fastf1.get_session(year, track_name, 'R')
                quali.load()
                race.load()
                
                # Get quali times
                quali_results = quali.results[['Abbreviation', 'Position', 'Q3', 'Q2', 'Q1']].copy()
                quali_results['BestQualiTime'] = quali_results['Q3'].fillna(
                    quali_results['Q2'].fillna(quali_results['Q1'])
                )
                
                # Get representative race pace (laps 8-30 to avoid start effects and tire deg)
                race_laps = race.laps
                representative_laps = race_laps[
                    (race_laps['LapNumber'] >= 8) &
                    (race_laps['LapNumber'] <= 30) &
                    (race_laps['TrackStatus'] == '1') &  # Green flag only
                    (~race_laps['PitOutTime'].notna()) &  # Not pit out lap
                    (~race_laps['PitInTime'].notna()) &   # Not pit in lap
                    (race_laps['Compound'].notna())       # Valid tire data
                ]
                
                # Group by driver and get median pace for each compound
                race_pace = []
                for driver in representative_laps['Driver'].unique():
                    driver_laps = representative_laps[representative_laps['Driver'] == driver]
                    
                    # Get pace by compound to normalize for strategy differences
                    for compound in driver_laps['Compound'].unique():
                        compound_laps = driver_laps[driver_laps['Compound'] == compound]
                        if len(compound_laps) >= 3:  # Minimum laps for reliable measurement
                            median_pace = compound_laps['LapTime'].median()
                            race_pace.append({
                                'Driver': driver,
                                'Compound': compound,
                                'MedianPace': median_pace,
                                'LapCount': len(compound_laps)
                            })
                
                race_pace_df = pd.DataFrame(race_pace)
                if len(race_pace_df) == 0:
                    continue
                
                race_pace_df['MedianPace_s'] = race_pace_df['MedianPace'].dt.total_seconds()
                
                # Merge with qualifying data
                merged = quali_results.merge(
                    race_pace_df, left_on='Abbreviation', right_on='Driver', how='inner'
                )
                
                merged = merged.dropna(subset=['BestQualiTime', 'MedianPace_s'])
                merged['QualiTime_s'] = merged['BestQualiTime'].dt.total_seconds()
                merged['GridPos'] = merged['Position'].astype(int)
                
                # Calculate pace delta accounting for compound differences
                # Normalize to a common compound baseline
                compound_corrections = {}
                for compound in merged['Compound'].unique():
                    compound_data = merged[merged['Compound'] == compound]
                    if len(compound_data) >= 3:
                        # Use median qualifying pace for this compound as baseline
                        baseline = compound_data['QualiTime_s'].median()
                        compound_corrections[compound] = baseline
                
                # Apply compound correction and calculate position penalty
                for idx, row in merged.iterrows():
                    if row['Compound'] in compound_corrections:
                        # Calculate expected race pace based on quali + compound correction
                        expected_race_pace = row['QualiTime_s'] + 0.5  # Normal quali->race delta
                        actual_race_pace = row['MedianPace_s']
                        position_penalty = actual_race_pace - expected_race_pace
                        
                        position_effects.append((row['GridPos'], position_penalty))
                        
            except Exception as e:
                print(f"  Warning: Could not process {year} position data - {e}")
                continue
        
        if position_effects:
            # Analyze position penalties
            position_penalties = {}
            for pos in range(1, 21):
                pos_deltas = [penalty for grid_pos, penalty in position_effects if grid_pos == pos]
                if len(pos_deltas) >= 3:  # Minimum samples for reliability
                    median_penalty = np.median(pos_deltas)
                    std_penalty = np.std(pos_deltas)
                    position_penalties[pos] = {
                        'penalty': max(0, median_penalty),  # Only positive penalties (traffic effects)
                        'std': std_penalty,
                        'sample_size': len(pos_deltas)
                    }
                    print(f"  P{pos}: +{median_penalty:.3f}s penalty (±{std_penalty:.3f}s, n={len(pos_deltas)})")
            
            print(f"  Total position penalty data points: {len(position_effects)}")
            return position_penalties
        else:
            print("  Warning: No position penalty data available")
            return {}
    
    def extract_tire_performance_soft_baseline(self, track_name, years):
        """Extract tire compound performance with SOFT as baseline"""
        print(f"\n2. Extracting tire performance (SOFT baseline)...")
        
        compound_data = {}
        
        for year in years:
            try:
                session = fastf1.get_session(year, track_name, 'R')
                session.load()
                
                laps = session.laps
                # Get clean representative laps
                clean_laps = laps[
                    (laps['TrackStatus'] == '1') &
                    (~laps['PitOutTime'].notna()) &
                    (~laps['PitInTime'].notna()) &
                    (laps['LapTime'].notna()) &
                    (laps['Compound'].notna())
                ]
                
                for compound in ['SOFT', 'MEDIUM', 'HARD']:
                    compound_laps = clean_laps[clean_laps['Compound'] == compound].copy()
                    
                    if len(compound_laps) < 15:
                        continue
                    
                    # Calculate stint lap and lap times
                    compound_laps['StintLap'] = compound_laps.groupby(['Driver', 'Stint']).cumcount() + 1
                    compound_laps['LapTime_s'] = compound_laps['LapTime'].dt.total_seconds()
                    
                    # Use laps 2-25 of stint for analysis (avoid out lap and extreme degradation)
                    analysis_laps = compound_laps[
                        (compound_laps['StintLap'] >= 2) & 
                        (compound_laps['StintLap'] <= 25)
                    ]
                    
                    if compound not in compound_data:
                        compound_data[compound] = []
                    
                    compound_data[compound].extend(
                        list(zip(analysis_laps['StintLap'], analysis_laps['LapTime_s']))
                    )
                    
            except Exception as e:
                print(f"  Warning: Could not process {year} tire data - {e}")
                continue
        
        # Analyze compound characteristics
        tire_params = {}
        
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            if compound not in compound_data or len(compound_data[compound]) < 30:
                print(f"  {compound}: Insufficient data (n={len(compound_data.get(compound, []))})")
                continue
                
            stint_laps = [x[0] for x in compound_data[compound]]
            lap_times = [x[1] for x in compound_data[compound]]
            
            # Fit linear degradation model
            X = np.array(stint_laps).reshape(-1, 1)
            y = np.array(lap_times)
            
            # Remove outliers (beyond 2.5 standard deviations)
            mean_time = np.mean(y)
            std_time = np.std(y)
            mask = np.abs(y - mean_time) <= 2.5 * std_time
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(y_clean) < 20:
                continue
            
            model = LinearRegression().fit(X_clean, y_clean)
            
            # Extract parameters
            base_time = model.intercept_
            degradation_rate = max(0, model.coef_[0])  # Ensure positive degradation
            r_squared = model.score(X_clean, y_clean)
            
            tire_params[compound] = {
                'base_time': base_time,
                'degradation_rate': degradation_rate,
                'r_squared': r_squared,
                'sample_size': len(compound_data[compound])
            }
            
            print(f"  {compound}: Base={base_time:.2f}s, Deg={degradation_rate:.4f}s/lap, R²={r_squared:.3f}, n={len(compound_data[compound])}")
        
        # Calculate offsets relative to SOFT compound
        if 'SOFT' in tire_params:
            soft_base = tire_params['SOFT']['base_time']
            for compound in tire_params:
                tire_params[compound]['offset'] = tire_params[compound]['base_time'] - soft_base
                
            print(f"  Offsets relative to SOFT:")
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                if compound in tire_params:
                    print(f"    {compound}: {tire_params[compound]['offset']:+.3f}s")
        else:
            print("  Warning: No SOFT compound data available for baseline")
        
        return tire_params
    
    def extract_driver_error_rates(self, track_name, years):
        """Extract driver error rates from lap time anomalies"""
        print(f"\n3. Extracting driver error rates...")
        
        error_data = {'dry': [], 'wet': []}
        
        for year in years:
            try:
                session = fastf1.get_session(year, track_name, 'R')
                session.load()
                
                laps = session.laps
                clean_laps = laps[
                    (laps['TrackStatus'] == '1') &
                    (~laps['PitOutTime'].notna()) &
                    (~laps['PitInTime'].notna()) &
                    (laps['LapTime'].notna()) &
                    (laps['Compound'].notna())
                ]
                
                # Analyze by weather condition
                dry_compounds = ['SOFT', 'MEDIUM', 'HARD']
                wet_compounds = ['INTERMEDIATE', 'WET']
                
                for weather_type, compounds in [('dry', dry_compounds), ('wet', wet_compounds)]:
                    weather_laps = clean_laps[clean_laps['Compound'].isin(compounds)]
                    
                    if len(weather_laps) < 50:
                        continue
                    
                    # Analyze each driver's lap time consistency
                    for driver in weather_laps['Driver'].unique():
                        driver_laps = weather_laps[weather_laps['Driver'] == driver]
                        
                        if len(driver_laps) < 10:
                            continue
                        
                        # Group by stint to account for tire degradation
                        for stint_num in driver_laps['Stint'].unique():
                            stint_laps = driver_laps[driver_laps['Stint'] == stint_num].copy()
                            stint_laps = stint_laps.sort_values('LapNumber')
                            
                            if len(stint_laps) < 5:
                                continue
                            
                            stint_laps['StintLap'] = range(1, len(stint_laps) + 1)
                            stint_laps['LapTime_s'] = stint_laps['LapTime'].dt.total_seconds()
                            
                            # Fit trend line to account for tire degradation
                            X = stint_laps['StintLap'].values.reshape(-1, 1)
                            y = stint_laps['LapTime_s'].values
                            
                            if len(y) >= 5:
                                model = LinearRegression().fit(X, y)
                                predicted = model.predict(X)
                                residuals = y - predicted
                                
                                # Define errors as laps significantly slower than trend
                                std_residual = np.std(residuals)
                                error_threshold = 2.0 * std_residual  # 2 sigma threshold
                                
                                error_laps = np.sum(residuals > error_threshold)
                                total_laps = len(residuals)
                                
                                if total_laps > 0:
                                    error_rate = error_laps / total_laps
                                    error_data[weather_type].append(error_rate)
                
            except Exception as e:
                print(f"  Warning: Could not process {year} error data - {e}")
                continue
        
        # Calculate error rate statistics
        error_rates = {}
        for weather_type in ['dry', 'wet']:
            if len(error_data[weather_type]) > 0:
                rates = error_data[weather_type]
                error_rates[weather_type] = {
                    'base_error_rate': np.median(rates),
                    'mean_error_rate': np.mean(rates),
                    'std_error_rate': np.std(rates),
                    'sample_size': len(rates)
                }
                print(f"  {weather_type.capitalize()} conditions: {np.median(rates):.4f} error rate "
                      f"(±{np.std(rates):.4f}, n={len(rates)} stints)")
            else:
                print(f"  {weather_type.capitalize()} conditions: No data")
        
        return error_rates
    
    def extract_drs_effectiveness(self, track_name, years):
        """Extract DRS effectiveness from lap time improvements"""
        print(f"\n4. Extracting DRS effectiveness...")
        
        drs_effects = []
        
        for year in years:
            try:
                session = fastf1.get_session(year, track_name, 'R')
                session.load()
                
                # Try to get DRS data (not all FastF1 versions have this)
                try:
                    laps = session.laps
                    
                    # Look for DRS-related columns or lap time improvements
                    # Since DRS data might not be directly available, we'll use proxy analysis
                    clean_laps = laps[
                        (laps['TrackStatus'] == '1') &
                        (~laps['PitOutTime'].notna()) &
                        (~laps['PitInTime'].notna()) &
                        (laps['LapTime'].notna()) &
                        (laps['Compound'].isin(['SOFT', 'MEDIUM', 'HARD']))  # Dry conditions only
                    ]
                    
                    # Analyze lap time improvements for cars running close together
                    # DRS is available when within 1 second of car ahead
                    for lap_num in clean_laps['LapNumber'].unique():
                        lap_data = clean_laps[clean_laps['LapNumber'] == lap_num]
                        lap_data = lap_data.sort_values('Position')
                        
                        if len(lap_data) < 5:
                            continue
                        
                        lap_data['LapTime_s'] = lap_data['LapTime'].dt.total_seconds()
                        
                        # Compare cars in similar positions with their baseline pace
                        for i in range(1, min(len(lap_data), 15)):  # Positions 2-15
                            current_car = lap_data.iloc[i]
                            
                            # Get this car's baseline pace (median of recent laps)
                            car_recent_laps = clean_laps[
                                (clean_laps['Driver'] == current_car['Driver']) &
                                (clean_laps['LapNumber'] >= lap_num - 5) &
                                (clean_laps['LapNumber'] < lap_num) &
                                (clean_laps['Compound'] == current_car['Compound'])
                            ]
                            
                            if len(car_recent_laps) >= 3:
                                baseline_pace = car_recent_laps['LapTime'].median().total_seconds()
                                current_pace = current_car['LapTime_s']
                                
                                # If significantly faster than baseline, could indicate DRS usage
                                pace_improvement = baseline_pace - current_pace
                                if 0.1 < pace_improvement < 1.0:  # Reasonable DRS effect range
                                    drs_effects.append(pace_improvement)
                
                except Exception as e:
                    print(f"    Could not extract detailed DRS data for {year}: {e}")
                    continue
                    
            except Exception as e:
                print(f"  Warning: Could not process {year} DRS data - {e}")
                continue
        
        if len(drs_effects) > 0:
            # Remove outliers
            drs_effects = np.array(drs_effects)
            q25, q75 = np.percentile(drs_effects, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            filtered_effects = drs_effects[(drs_effects >= lower_bound) & (drs_effects <= upper_bound)]
            
            drs_effectiveness = {
                'mean_advantage': np.mean(filtered_effects),
                'median_advantage': np.median(filtered_effects),
                'std_advantage': np.std(filtered_effects),
                'sample_size': len(filtered_effects),
                'usage_probability': 0.3  # Estimate based on typical following distances
            }
            
            print(f"  DRS advantage: {np.median(filtered_effects):.3f}s "
                  f"(±{np.std(filtered_effects):.3f}s, n={len(filtered_effects)})")
            
        else:
            # Use theoretical estimates based on track characteristics
            print("  Using theoretical DRS estimates (no direct data available)")
            drs_effectiveness = {
                'mean_advantage': 0.25,  # Typical DRS advantage
                'median_advantage': 0.25,
                'std_advantage': 0.1,
                'sample_size': 0,
                'usage_probability': 0.3
            }
            
        return drs_effectiveness
    
    def generate_parameter_file(self, parameters, track_name):
        """Generate a Python file with extracted parameters"""
        filename = f"{track_name.lower().replace(' ', '_')}_targeted_parameters.py"
        
        with open(filename, 'w') as f:
            f.write(f'"""\n')
            f.write(f'Extracted F1 simulation parameters for {track_name}\n')
            f.write(f'Targeted extraction: Position penalties, tire performance, driver errors, DRS\n')
            f.write(f'Generated automatically from historical FastF1 data\n')
            f.write(f'"""\n\n')
            f.write(f'import numpy as np\n\n')
            
            # Write each parameter section
            for section, data in parameters.items():
                f.write(f'# {section.upper().replace("_", " ")}\n')
                f.write(f'{section.upper()} = {repr(data)}\n\n')
            
            # Write convenience functions
            f.write('# CONVENIENCE FUNCTIONS\n\n')
            
            f.write('def get_position_penalty(position):\n')
            f.write('    """Get traffic/dirty air penalty for grid position"""\n')
            f.write('    if position in POSITION_PENALTIES:\n')
            f.write('        return POSITION_PENALTIES[position]["penalty"]\n')
            f.write('    else:\n')
            f.write('        # Extrapolate for positions beyond data\n')
            f.write('        if position <= 20:\n')
            f.write('            return 0.05 * (position - 1)  # Linear approximation\n')
            f.write('        else:\n')
            f.write('            return 1.0  # High penalty for back of grid\n\n')
            
            f.write('def get_tire_offset(compound):\n')
            f.write('    """Get tire compound offset relative to SOFT"""\n')
            f.write('    return TIRE_PERFORMANCE.get(compound, {}).get("offset", 0.0)\n\n')
            
            f.write('def get_tire_degradation_rate(compound):\n')
            f.write('    """Get tire degradation rate in s/lap"""\n')
            f.write('    return TIRE_PERFORMANCE.get(compound, {}).get("degradation_rate", 0.08)\n\n')
            
            f.write('def get_driver_error_rate(weather_condition="dry"):\n')
            f.write('    """Get driver error probability per lap"""\n')
            f.write('    return DRIVER_ERROR_RATES.get(weather_condition, {}).get("base_error_rate", 0.01)\n\n')
            
            f.write('def get_drs_advantage():\n')
            f.write('    """Get DRS time advantage in seconds"""\n')
            f.write('    mean_adv = DRS_EFFECTIVENESS.get("median_advantage", 0.25)\n')
            f.write('    std_adv = DRS_EFFECTIVENESS.get("std_advantage", 0.1)\n')
            f.write('    return max(0.1, np.random.normal(mean_adv, std_adv))\n\n')
            
            f.write('def get_drs_usage_probability():\n')
            f.write('    """Get probability of being in DRS range"""\n')
            f.write('    return DRS_EFFECTIVENESS.get("usage_probability", 0.3)\n\n')
        
        print(f"\nTargeted parameter file saved as: {filename}")
        
        # Create summary report
        self.create_summary_report(parameters, track_name)
    
    def create_summary_report(self, parameters, track_name):
        """Create a summary report of extracted parameters"""
        print(f"\n" + "="*60)
        print(f"TARGETED PARAMETER EXTRACTION SUMMARY - {track_name.upper()}")
        print("="*60)
        
        # Data quality assessment
        total_samples = 0
        high_quality_sections = 0
        
        for section, data in parameters.items():
            print(f"\n{section.upper().replace('_', ' ')}:")
            
            if section == 'position_penalties':
                if data:
                    positions_with_data = len(data)
                    total_penalty_samples = sum(pos_data['sample_size'] for pos_data in data.values())
                    print(f"  Positions covered: {positions_with_data}/20")
                    print(f"  Total samples: {total_penalty_samples}")
                    total_samples += total_penalty_samples
                    if total_penalty_samples > 50:
                        high_quality_sections += 1
                else:
                    print("  No data extracted")
                    
            elif section == 'tire_performance':
                if data:
                    compounds_with_data = len(data)
                    for compound, comp_data in data.items():
                        sample_size = comp_data.get('sample_size', 0)
                        r_squared = comp_data.get('r_squared', 0)
                        print(f"  {compound}: {sample_size} samples, R²={r_squared:.3f}")
                        total_samples += sample_size
                    if compounds_with_data >= 2 and total_samples > 100:
                        high_quality_sections += 1
                else:
                    print("  No tire data extracted")
                    
            elif section == 'driver_error_rates':
                for condition in ['dry', 'wet']:
                    if condition in data:
                        sample_size = data[condition].get('sample_size', 0)
                        error_rate = data[condition].get('base_error_rate', 0)
                        print(f"  {condition.capitalize()}: {error_rate:.4f} rate, {sample_size} samples")
                        total_samples += sample_size
                if any(data.get(cond, {}).get('sample_size', 0) > 20 for cond in ['dry', 'wet']):
                    high_quality_sections += 1
                    
            elif section == 'drs_effectiveness':
                sample_size = data.get('sample_size', 0)
                advantage = data.get('median_advantage', 0)
                print(f"  DRS advantage: {advantage:.3f}s, {sample_size} samples")
                total_samples += sample_size
                if sample_size > 10:
                    high_quality_sections += 1
        
        data_quality = high_quality_sections / len(parameters) * 100
        
        print(f"\nOVERALL QUALITY ASSESSMENT:")
        print(f"Data Quality Score: {data_quality:.1f}% ({high_quality_sections}/{len(parameters)} sections with good data)")
        print(f"Total Data Points: {total_samples}")
        
        if data_quality >= 75:
            print("Status: ✓ High quality extraction - reliable for simulation")
        elif data_quality >= 50:
            print("Status: ⚠ Moderate quality - usable with caution")
        else:
            print("Status: ❌ Low quality - consider using default values")

def main():
    """Main function to run targeted parameter extraction"""
    import sys
    
    if len(sys.argv) > 1:
        track_name = ' '.join(sys.argv[1:])
    else:
        track_name = 'Dutch Grand Prix'  # Default
    
    print("F1 TARGETED PARAMETER EXTRACTOR")
    print("="*40)
    print("Extracting: Position penalties, tire performance, driver errors, DRS effectiveness")
    print("This focused extraction will be faster and more reliable.")
    
    extractor = F1ParameterExtractor()
    
    try:
        parameters = extractor.extract_targeted_parameters(track_name)
        
        print(f"\n✓ Targeted parameter extraction complete!")
        print(f"✓ Parameter file generated: {track_name.lower().replace(' ', '_')}_targeted_parameters.py")
        print(f"✓ Use these parameters in your simulation for improved accuracy")
        
    except Exception as e:
        print(f"\nError during extraction: {e}")
        print("This could be due to:")
        print("- Network connectivity issues")
        print("- Missing race data for specified years")
        print("- FastF1 API changes")
        print("- Invalid track name")

if __name__ == "__main__":
    main()