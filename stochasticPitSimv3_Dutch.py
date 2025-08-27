import fastf1
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Import extracted targeted parameters - run F1_Parameter_Extractor.py first to generate this file
try:
    from dutch_grand_prix_targeted_parameters import *
    PARAMETERS_LOADED = True
    print("Using extracted targeted parameters from dutch_grand_prix_targeted_parameters.py")
except ImportError:
    print("Warning: dutch_grand_prix_targeted_parameters.py not found. Run F1_Parameter_Extractor.py first.")
    print("Using fallback default parameters.")
    PARAMETERS_LOADED = False

# Fallback parameters if extraction file not available
if not PARAMETERS_LOADED:
    POSITION_PENALTIES = {i: {'penalty': 0.12 * (i-1)} for i in range(1, 21)}
    TIRE_PERFORMANCE = {
        'SOFT': {'offset': 0.0, 'degradation_rate': 0.18},  # SOFT as baseline
        'MEDIUM': {'offset': 0.35, 'degradation_rate': 0.09},
        'HARD': {'offset': 0.65, 'degradation_rate': 0.05}
    }
    DRIVER_ERROR_RATES = {
        'dry': {'base_error_rate': 0.012},
        'wet': {'base_error_rate': 0.030}
    }
    DRS_EFFECTIVENESS = {
        'median_advantage': 0.25,
        'usage_probability': 0.30
    }

# Zandvoort-specific parameters
ZANDVOORT_PARAMS = {
    'base_pace': 71.0,  # Zandvoort lap time
    'num_laps': 72,     # Dutch GP distance
    'rain_probability': 0.30,   # 30% chance of rain
    'sc_probability': 0.67,     # 67% SC probability
    'vsc_probability': 0.67,    # 67% VSC probability  
    'pit_time_loss': 21.52,     # 21.52 seconds pit loss
    'fuel_consumption_rate': 110 / 72,  # 110 kg max starting weight / num laps
    'fuel_effect_per_kg': 0.035  # Standard F1 fuel effect
}

# load 2024 Dutch GP data for tire deg modeling
session = fastf1.get_session(2024, 'Dutch Grand Prix', 'R')
session.load()

# load 2025 qualifying results for grid positions
quali = fastf1.get_session(2025, 'Dutch Grand Prix', 'Q')
quali.load()

# process race data
laps = session.laps
stints = laps[["Driver", "Stint", "Compound", "LapNumber", "LapTime"]].copy()
stints["LapTime_s"] = stints["LapTime"].dt.total_seconds()
stints.dropna(subset=["LapTime_s"], inplace=True)
stints["StintLap"] = stints.groupby(["Driver", "Stint"]).cumcount() + 1

# process qualifying results for grid analysis
quali_results = quali.results[['Abbreviation', 'Position']].copy()
quali_results = quali_results.dropna()
quali_results['GridPosition'] = quali_results['Position'].astype(int)

print("Grid positions from 2025 Dutch GP qualifying:")
print(quali_results[['Abbreviation', 'GridPosition']].head(10))

# tire deg model using extracted base pace
def build_tire_model(compound_data):
    if len(compound_data) < 5:
        return None
    
    x = compound_data["StintLap"].values
    y = compound_data["LapTime_s"].values

    def model(x, y=None):
        # Use Zandvoort base pace
        alpha = numpyro.sample("alpha", dist.Normal(ZANDVOORT_PARAMS['base_pace'], 4))
        # Use extracted degradation rate if available
        compound = compound_data['Compound'].iloc[0] if 'Compound' in compound_data.columns else 'MEDIUM'
        expected_deg = get_tire_degradation_rate(compound)
        beta = numpyro.sample("beta", dist.Normal(expected_deg, 0.02))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))
        mu = alpha + beta * x
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
    mcmc.run(random.PRNGKey(0), x, y)
    return mcmc

# build models with compound-specific deg
compound_models = {}
for compound in stints["Compound"].unique():
    data = stints[stints["Compound"] == compound].copy()
    data["Compound"] = compound  # Ensure compound info is available
    if len(data) > 5:
        mcmc = build_tire_model(data)
        compound_models[compound] = mcmc

# tire performance using extracted parameters
def get_tire_performance(compound, lap_in_stint, base_pace=None, weather='dry', track_evolution=0):
    """Calculate lap time based on compound, stint lap, weather, and track evolution"""
    
    if base_pace is None:
        base_pace = ZANDVOORT_PARAMS['base_pace']
    
    if weather == 'dry':
        # Use extracted compound offsets and degradation rates (SOFT baseline)
        compound_offset = get_tire_offset(compound)
        deg_rate = get_tire_degradation_rate(compound)
        
        base_time = base_pace + compound_offset
        degradation = deg_rate * lap_in_stint
        
    else:  # wet weather
        compound_offsets = {
            'INTERMEDIATE': 0.0,  # baseline for wet (no WET tire)
        }
        
        deg_rates = {
            'INTERMEDIATE': 0.03,  # Slightly higher deg in wet
        }
        
        # base wet pace 
        base_time = base_pace + 9 + compound_offsets.get(compound, 0)
        degradation = deg_rates.get(compound, 0.03) * lap_in_stint
    
    # track evolution (negative = getting faster)
    base_time += track_evolution
    
    # non-linear deg for long stints (more punishing at Zandvoort)
    if lap_in_stint > 25:
        degradation += 0.04 * (lap_in_stint - 25) ** 1.4
    
    return base_time + degradation

# weather simulation using Zandvoort parameters
def generate_weather_conditions(num_laps, rain_probability=None):
    """Generate weather conditions throughout the race"""
    if rain_probability is None:
        rain_probability = ZANDVOORT_PARAMS['rain_probability']
    
    weather_laps = ['dry'] * num_laps
    
    if np.random.rand() < rain_probability:
        # rain scenarios (adjusted for Zandvoort characteristics)
        rain_scenarios = [
            'early_shower',    # rain laps 5-15
            'mid_race_rain',   # rain laps 20-35
            'late_drama',      # rain laps 40-50
            'brief_shower'     # single short shower
        ]
        
        scenario = np.random.choice(rain_scenarios)
        
        if scenario == 'early_shower':
            rain_start = np.random.choice(range(3, 8))
            rain_end = rain_start + np.random.choice(range(6, 12))
            for lap in range(rain_start, min(rain_end, num_laps)):
                weather_laps[lap] = 'wet'
                
        elif scenario == 'mid_race_rain':
            rain_start = np.random.choice(range(18, 25))
            rain_end = rain_start + np.random.choice(range(10, 16))
            for lap in range(rain_start, min(rain_end, num_laps)):
                weather_laps[lap] = 'wet'
                
        elif scenario == 'late_drama':
            rain_start = np.random.choice(range(35, 45))
            for lap in range(rain_start, num_laps):
                weather_laps[lap] = 'wet'
                
        elif scenario == 'brief_shower':
            shower_start = np.random.choice(range(10, num_laps-8))
            shower_length = np.random.choice(range(3, 6))
            for lap in range(shower_start, min(shower_start + shower_length, num_laps)):
                weather_laps[lap] = 'wet'
    
    return weather_laps

# safety car simulation using Zandvoort parameters
def generate_sc_laps(num_laps, weather_conditions):
    """Generate safety car periods based on track and weather"""
    sc_laps = set()
    
    # Use Zandvoort SC probability (67%)
    base_sc_prob = ZANDVOORT_PARAMS['sc_probability']
    
    if np.random.rand() < base_sc_prob:
        # determine SC cause and timing
        wet_laps = sum(1 for w in weather_conditions if w == 'wet')
        
        if wet_laps > 0:
            # weather-related incident
            wet_lap_indices = [i for i, w in enumerate(weather_conditions) if w == 'wet']
            if wet_lap_indices:
                sc_start = np.random.choice(wet_lap_indices[:len(wet_lap_indices)//2])
                sc_duration = np.random.choice(range(3, 7))
        else:
            # regular incident
            sc_start = np.random.choice(range(8, num_laps-5))
            sc_duration = np.random.choice(range(2, 5))
        
        sc_laps.update(range(sc_start, min(sc_start + sc_duration, num_laps)))
    
    return sc_laps

# position-based performance model using extracted penalties
def get_position_penalty(grid_pos, current_lap, total_laps):
    """Calculate time penalty based on starting grid position and traffic"""
    
    # Use extracted position penalties
    if PARAMETERS_LOADED:
        base_penalty = get_position_penalty_extracted(grid_pos)
    else:
        # Fallback penalties (harder to overtake at Zandvoort)
        penalty_map = {
            1: 0.0, 2: 0.12, 3: 0.18, 4: 0.24, 5: 0.30,
            6: 0.36, 7: 0.42, 8: 0.48, 9: 0.54, 10: 0.60
        }
        base_penalty = penalty_map.get(min(grid_pos, 10), 0.60 + (grid_pos - 10) * 0.06)
    
    # traffic penalty reduces more slowly at Zandvoort (harder to spread out)
    traffic_factor = max(0.4, 1.0 - (current_lap / total_laps) * 0.6)
    
    return base_penalty * traffic_factor

# race simulation using extracted and Zandvoort parameters
def simulate_race(strategy, grid_position, compound_models, num_laps=None,
                 base_pace=None, rain_probability=None, car_performance_factor=1.0):
    """Simulate a complete race with weather and position effects"""
    
    if num_laps is None:
        num_laps = ZANDVOORT_PARAMS['num_laps']
    if base_pace is None:
        base_pace = ZANDVOORT_PARAMS['base_pace']
    if rain_probability is None:
        rain_probability = ZANDVOORT_PARAMS['rain_probability']
    
    race_time = 0
    current_lap = 1
    pit_time_loss = ZANDVOORT_PARAMS['pit_time_loss']
    current_position = grid_position
    fuel_load = 110  # starting fuel load in kg
    fuel_consumption_rate = ZANDVOORT_PARAMS['fuel_consumption_rate']
    
    # generate race conditions
    weather_conditions = generate_weather_conditions(num_laps, rain_probability)
    sc_laps = generate_sc_laps(num_laps, weather_conditions)
    
    # track evolution (gets faster over time in dry conditions)
    track_evolution_rate = -0.0025
    
    # VSC simulation using Zandvoort probability (67%)
    vsc_laps = set()
    if np.random.rand() < ZANDVOORT_PARAMS['vsc_probability']:
        vsc_start = np.random.choice(range(10, num_laps-5))
        vsc_laps.update(range(vsc_start, vsc_start + 2))
    
    for stint_idx, stint in enumerate(strategy):
        comp = stint["compound"]
        stint_len = stint["laps"]
        
        # adjust stint length if it would exceed race distance
        remaining_laps = num_laps - current_lap + 1
        stint_len = min(stint_len, remaining_laps)
        
        for lap_in_stint in range(1, stint_len + 1):
            if current_lap > num_laps:
                break
            
            # current weather
            current_weather = weather_conditions[current_lap - 1]
            
            # track evolution
            track_evolution = track_evolution_rate * current_lap if current_weather == 'dry' else 0
            
            # base lap time from tire model
            lap_time = get_tire_performance(comp, lap_in_stint, base_pace, 
                                          current_weather, track_evolution)
            
            # car performance factor 
            lap_time *= car_performance_factor
            
            # fuel effect using Zandvoort parameters
            fuel_effect = -(fuel_load * ZANDVOORT_PARAMS['fuel_effect_per_kg'])
            lap_time += fuel_effect
            fuel_load = max(5, fuel_load - fuel_consumption_rate)
            
            # position-based penalty using extracted data
            position_penalty = get_position_penalty(current_position, current_lap, num_laps)
            lap_time += position_penalty
            
            # DRS effect using extracted effectiveness
            if current_weather == 'dry' and current_position > 1:
                drs_usage_prob = get_drs_usage_probability()
                # Adjust for position (harder to follow closely at back)
                position_adjusted_prob = drs_usage_prob * max(0.3, 1.0 - (current_position * 0.05))
                
                if np.random.rand() < position_adjusted_prob:
                    drs_advantage = get_drs_advantage()
                    lap_time -= drs_advantage
            
            # tire temperature effects (more pronounced at Zandvoort)
            if current_weather == 'dry':
                if comp == 'SOFT' and lap_in_stint > 8:
                    lap_time += min(0.6, (lap_in_stint - 8) * 0.025)
                elif comp == 'HARD' and lap_in_stint < 6:
                    lap_time += max(0, (6 - lap_in_stint) * 0.12)
            
            # weather transition penalties
            if current_lap > 1:
                prev_weather = weather_conditions[current_lap - 2]
                if prev_weather != current_weather:
                    if current_weather == 'wet' and comp in ['SOFT', 'MEDIUM', 'HARD']:
                        lap_time += np.random.uniform(3.5, 9.0)
                    elif current_weather == 'dry' and comp in ['INTERMEDIATE']:
                        lap_time += np.random.uniform(2.0, 4.0)
            
            # driver error using extracted rates
            base_error_rate = get_driver_error_rate(current_weather)
            # Increase with stint length
            error_probability = base_error_rate + (lap_in_stint * 0.0005)
            
            if np.random.rand() < error_probability:
                lap_time += np.random.uniform(1.2, 3.5)
            
            # random variation
            lap_time += np.random.normal(0, 0.35)
            
            # safety car effect
            if current_lap in sc_laps:
                lap_time *= 1.35
            elif current_lap in vsc_laps:
                lap_time *= 1.18
            
            race_time += lap_time
            current_lap += 1
        
        # pit stop execution
        if stint_idx < len(strategy) - 1:
            pit_execution = np.random.normal(pit_time_loss, 1.2)
            
            # reduced pit loss during SC/VSC
            if any(lap in sc_laps for lap in range(max(1, current_lap-2), current_lap+1)):
                pit_execution *= 0.20  # Better SC advantage
            elif any(lap in vsc_laps for lap in range(max(1, current_lap-2), current_lap+1)):
                pit_execution *= 0.55
            
            # tire change complexity
            prev_compound = strategy[stint_idx]["compound"]
            next_compound = strategy[stint_idx + 1]["compound"]
            if prev_compound != next_compound:
                pit_execution += 0.4
            
            race_time += max(13, pit_execution)
            
            # strategic position changes (harder to gain at Zandvoort)
            strategy_aggressiveness = len([s for s in strategy if s["compound"] in ["SOFT", "INTERMEDIATE"]])
            undercut_probability = 0.25 if stint_idx == 0 else 0.12
            
            position_change = 0
            if np.random.rand() < undercut_probability:
                position_change = np.random.choice([-2, -1], p=[0.25, 0.75])
            else:
                position_change = np.random.choice([-1, 0, 1, 2], p=[0.10, 0.50, 0.30, 0.10])
            
            if strategy_aggressiveness >= 2:
                position_change += np.random.choice([-1, 0], p=[0.3, 0.7])
            
            current_position = max(1, min(20, current_position + position_change))
    
    return race_time, current_position

# tire strategies (from historical data)
dry_strategies = {
    "1-stop (M-H)": [
        {"compound": "MEDIUM", "laps": 30},
        {"compound": "HARD", "laps": 42}
    ],
    "2-stop (S-M-H)": [
        {"compound": "SOFT", "laps": 18},
        {"compound": "MEDIUM", "laps": 20},
        {"compound": "HARD", "laps": 34}
    ],
    "2-stop (M-H-S)": [
        {"compound": "MEDIUM", "laps": 28},
        {"compound": "HARD", "laps": 26},
        {"compound": "SOFT", "laps": 18}
    ],
    "1-stop (H-M)": [
        {"compound": "HARD", "laps": 40},
        {"compound": "MEDIUM", "laps": 32}
    ],
    "2-stop (S-H-M)": [
        {"compound": "SOFT", "laps": 10},
        {"compound": "HARD", "laps": 32},
        {"compound": "MEDIUM", "laps": 30}
    ],
    "2-stop (S-M-S)": [
        {"compound": "SOFT", "laps": 22},
        {"compound": "MEDIUM", "laps": 30},
        {"compound": "SOFT", "laps": 20}
    ],
    "2-stop (S-H-S)": [
        {"compound": "SOFT", "laps": 18},
        {"compound": "HARD", "laps": 35},
        {"compound": "SOFT", "laps": 19}
    ]
}

# wet strategies (NO WET TIRE - only INTERMEDIATE)
wet_strategies = {
    "Wet Conservative": [
        {"compound": "INTERMEDIATE", "laps": 24},
        {"compound": "INTERMEDIATE", "laps": 24},
        {"compound": "MEDIUM", "laps": 24}
    ],
    "Wet Aggressive": [
        {"compound": "INTERMEDIATE", "laps": 20},
        {"compound": "INTERMEDIATE", "laps": 20},
        {"compound": "INTERMEDIATE", "laps": 16},
        {"compound": "SOFT", "laps": 16}
    ],
    "Gamble on Dry": [
        {"compound": "MEDIUM", "laps": 36},
        {"compound": "HARD", "laps": 36}
    ]
}

# combine strategies
all_strategies = {**dry_strategies, **wet_strategies}

# Convenience functions for extracted parameters
def get_tire_offset(compound):
    if PARAMETERS_LOADED:
        return TIRE_PERFORMANCE.get(compound, {}).get('offset', 0.0)
    else:
        # Fallback offsets (SOFT baseline)
        offsets = {'SOFT': 0.0, 'MEDIUM': 0.35, 'HARD': 0.65}
        return offsets.get(compound, 0.0)

def get_tire_degradation_rate(compound):
    if PARAMETERS_LOADED:
        return TIRE_PERFORMANCE.get(compound, {}).get('degradation_rate', 0.08)
    else:
        # Fallback degradation rates
        rates = {'SOFT': 0.18, 'MEDIUM': 0.09, 'HARD': 0.05}
        return rates.get(compound, 0.08)

def get_position_penalty_extracted(position):
    if PARAMETERS_LOADED:
        return POSITION_PENALTIES.get(position, {}).get('penalty', 0.05 * (position - 1))
    else:
        return 0.05 * (position - 1)

def get_driver_error_rate(weather_condition='dry'):
    if PARAMETERS_LOADED:
        return DRIVER_ERROR_RATES.get(weather_condition, {}).get('base_error_rate', 0.01)
    else:
        return 0.030 if weather_condition == 'wet' else 0.012

def get_drs_advantage():
    if PARAMETERS_LOADED:
        mean_adv = DRS_EFFECTIVENESS.get('median_advantage', 0.25)
        std_adv = DRS_EFFECTIVENESS.get('std_advantage', 0.1)
        return max(0.1, np.random.normal(mean_adv, std_adv))
    else:
        return 0.25

def get_drs_usage_probability():
    if PARAMETERS_LOADED:
        return DRS_EFFECTIVENESS.get('usage_probability', 0.3)
    else:
        return 0.30

# points system
def get_f1_points(position):
    """Return F1 championship points for finishing position"""
    points_map = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    return points_map.get(position, 0)

# Monte Carlo simulation with car performance differentiation
def run_monte_carlo_with_grid(strategies, compound_models, grid_positions, num_sims=500):
    """Run simulation across different grid positions with car performance factors"""
    results = {}
    
    # car performance factors based on 2025 grid positions (approximation)
    car_performance_map = {
        1: 0.98,   2: 0.985,  3: 0.99,   4: 0.995,  5: 1.00,   
        6: 1.005,  7: 1.01,   8: 1.015,  9: 1.02,   10: 1.025,
        11: 1.03,  12: 1.035, 13: 1.04,  14: 1.045, 15: 1.05,  
    }
    
    for grid_pos in grid_positions:
        print(f"\nSimulating from grid position {grid_pos}...")
        pos_results = {name: {'times': [], 'final_positions': [], 'points': []} for name in strategies.keys()}
        
        car_factor = car_performance_map.get(grid_pos, 1.0 + (grid_pos - 15) * 0.005)
        
        for _ in trange(num_sims, desc=f"P{grid_pos}"):
            # Use Zandvoort parameters
            base_pace = np.random.normal(ZANDVOORT_PARAMS['base_pace'], 0.5)
            sim_rain_prob = ZANDVOORT_PARAMS['rain_probability']
            
            for name, strat in strategies.items():
                race_time, final_pos = simulate_race(
                    strat, grid_pos, compound_models, 
                    base_pace=base_pace, rain_probability=sim_rain_prob,
                    car_performance_factor=car_factor
                )
                
                points = get_f1_points(final_pos)
                
                pos_results[name]['times'].append(race_time)
                pos_results[name]['final_positions'].append(final_pos)
                pos_results[name]['points'].append(points)
        
        results[grid_pos] = pos_results
    
    return results

# visualization
def plot_strategy_analysis(results, grid_positions):
    """Create comprehensive strategy analysis plots"""
    
    rain_prob = ZANDVOORT_PARAMS['rain_probability']
    
    # Figure 1: Race time distributions
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))
    fig1.suptitle(f'Dutch GP Strategy Analysis - Race Time Distributions ({rain_prob:.0%} Rain Probability)', fontsize=14)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_strategies)))
    strategy_colors = dict(zip(all_strategies.keys(), colors))
    
    for i, grid_pos in enumerate([1, 5, 10]):
        ax = axes1[i]
        for strategy_name in all_strategies.keys():
            times = results[grid_pos][strategy_name]['times']
            ax.hist(times, bins=25, alpha=0.6, label=strategy_name, 
                   color=strategy_colors[strategy_name])
        
        ax.set_title(f'Race Time Distribution - Grid P{grid_pos}')
        ax.set_xlabel('Race Time (s)')
        ax.set_ylabel('Frequency')
        if i == 2:  
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Final position distributions
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle(f'Dutch GP Strategy Analysis - Final Position Distributions ({rain_prob:.0%} Rain Probability)', fontsize=14)
    
    for i, grid_pos in enumerate([1, 5, 10]):
        ax = axes2[i]
        
        final_pos_data = []
        strategy_names = []
        
        for strategy_name in all_strategies.keys():
            positions = results[grid_pos][strategy_name]['final_positions']
            final_pos_data.extend(positions)
            strategy_names.extend([strategy_name] * len(positions))
        
        df_pos = pd.DataFrame({'Strategy': strategy_names, 'Final_Position': final_pos_data})
        
        sns.boxplot(data=df_pos, x='Strategy', y='Final_Position', ax=ax)
        ax.set_title(f'Final Positions - Grid P{grid_pos}')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Final Position')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        ax.set_ylim(0, 20)
    
    plt.tight_layout()
    plt.show()
    
    # Figure 3: Strategy effectiveness heatmap
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
    
    heatmap_data = []
    strategy_names = list(all_strategies.keys())
    
    for strategy in strategy_names:
        row = []
        for grid_pos in [1, 3, 5, 8, 10, 15]:
            avg_pos = np.mean(results[grid_pos][strategy]['final_positions'])
            row.append(avg_pos)
        heatmap_data.append(row)
    
    heatmap_array = np.array(heatmap_data)
    
    im = ax3.imshow(heatmap_array, cmap='RdYlGn_r', aspect='auto')
    
    ax3.set_xticks(range(len([1, 3, 5, 8, 10, 15])))
    ax3.set_xticklabels([f'P{pos}' for pos in [1, 3, 5, 8, 10, 15]])
    ax3.set_yticks(range(len(strategy_names)))
    ax3.set_yticklabels(strategy_names)
    
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Average Final Position', rotation=270, labelpad=20)
    
    for i in range(len(strategy_names)):
        for j in range(len([1, 3, 5, 8, 10, 15])):
            text = ax3.text(j, i, f'{heatmap_array[i, j]:.1f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax3.set_title('Strategy Effectiveness by Grid Position\n(Lower numbers = better final position)', fontsize=14)
    ax3.set_xlabel('Starting Grid Position')
    ax3.set_ylabel('Strategy')
    
    plt.tight_layout()
    plt.show()

# run simulations for key grid positions
key_positions = [1, 3, 5, 8, 10, 15]
print("Running Monte Carlo simulation for Dutch GP...")
print(f"Parameters: {ZANDVOORT_PARAMS['rain_probability']:.0%} rain, {ZANDVOORT_PARAMS['sc_probability']:.0%} SC, {ZANDVOORT_PARAMS['vsc_probability']:.0%} VSC")

results = run_monte_carlo_with_grid(all_strategies, compound_models, key_positions, num_sims=300)

# generate analysis
plot_strategy_analysis(results, key_positions)

# results summary
print("\n" + "="*80)
print("DUTCH GP STRATEGY ANALYSIS SUMMARY")
print("="*80)
print(f"Parameters used: {'Extracted + Zandvoort-specific' if PARAMETERS_LOADED else 'Fallback + Zandvoort-specific'}")
print(f"Rain: {ZANDVOORT_PARAMS['rain_probability']:.0%}, SC: {ZANDVOORT_PARAMS['sc_probability']:.0%}, VSC: {ZANDVOORT_PARAMS['vsc_probability']:.0%}")
print(f"Pit time loss: {ZANDVOORT_PARAMS['pit_time_loss']:.1f}s, Fuel: {ZANDVOORT_PARAMS['fuel_consumption_rate']:.2f} kg/lap")

for grid_pos in key_positions:
    print(f"\n GRID POSITION {grid_pos}")
    print("-" * 50)
    
    strategy_summary = []
    for strategy_name, data in results[grid_pos].items():
        times = np.array(data['times'])
        positions = np.array(data['final_positions'])
        points = np.array(data['points'])
        
        strategy_summary.append({
            'Strategy': strategy_name,
            'Avg Time': f"{np.mean(times):.1f}s",
            'Avg Final Pos': f"{np.mean(positions):.1f}",
            'Avg Points': f"{np.mean(points):.1f}",
            'Points Finish %': f"{np.mean(points > 0)*100:.1f}%",
            'Top 5 Finish %': f"{np.mean(positions <= 5)*100:.1f}%",
            'Podium %': f"{np.mean(positions <= 3)*100:.1f}%" if grid_pos <= 10 else "N/A",
            'Win %': f"{np.mean(positions == 1)*100:.1f}%" if grid_pos <= 5 else "N/A"
        })
    
    summary_df = pd.DataFrame(strategy_summary)
    summary_df = summary_df.sort_values('Avg Points', ascending=False)
    print(summary_df.to_string(index=False))

# Display parameter summary
print("\n" + "="*80)
print("PARAMETER SUMMARY")
print("="*80)

print(f"ZANDVOORT-SPECIFIC PARAMETERS:")
print(f"Base Pace: {ZANDVOORT_PARAMS['base_pace']:.1f}s")
print(f"Race Distance: {ZANDVOORT_PARAMS['num_laps']} laps")
print(f"Rain Probability: {ZANDVOORT_PARAMS['rain_probability']:.0%}")
print(f"Safety Car Probability: {ZANDVOORT_PARAMS['sc_probability']:.0%}")
print(f"VSC Probability: {ZANDVOORT_PARAMS['vsc_probability']:.0%}")
print(f"Pit Stop Time Loss: {ZANDVOORT_PARAMS['pit_time_loss']:.2f}s")
print(f"Fuel Consumption: {ZANDVOORT_PARAMS['fuel_consumption_rate']:.3f} kg/lap ({109}/{ZANDVOORT_PARAMS['num_laps']} laps)")
print(f"Fuel Effect: {ZANDVOORT_PARAMS['fuel_effect_per_kg']:.3f}s/kg")

if PARAMETERS_LOADED:
    print(f"\nEXTRACTED PARAMETERS:")
    print(f"Tire Compound Offsets (vs SOFT baseline):")
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        if compound in TIRE_PERFORMANCE:
            offset = TIRE_PERFORMANCE[compound].get('offset', 0)
            deg_rate = TIRE_PERFORMANCE[compound].get('degradation_rate', 0)
            sample_size = TIRE_PERFORMANCE[compound].get('sample_size', 0)
            print(f"  {compound}: {offset:+.3f}s offset, {deg_rate:.4f}s/lap deg (n={sample_size})")
    
    print(f"\nPosition Penalties (sample):")
    for pos in [1, 3, 5, 8, 10]:
        if pos in POSITION_PENALTIES:
            penalty = POSITION_PENALTIES[pos]['penalty']
            sample_size = POSITION_PENALTIES[pos].get('sample_size', 0)
            print(f"  P{pos}: +{penalty:.3f}s penalty (n={sample_size})")
    
    print(f"\nDriver Error Rates:")
    for condition in ['dry', 'wet']:
        if condition in DRIVER_ERROR_RATES:
            error_rate = DRIVER_ERROR_RATES[condition]['base_error_rate']
            sample_size = DRIVER_ERROR_RATES[condition].get('sample_size', 0)
            print(f"  {condition.capitalize()}: {error_rate:.4f} base rate (n={sample_size})")
    
    print(f"\nDRS Effectiveness:")
    drs_advantage = DRS_EFFECTIVENESS.get('median_advantage', 0)
    drs_usage = DRS_EFFECTIVENESS.get('usage_probability', 0)
    drs_samples = DRS_EFFECTIVENESS.get('sample_size', 0)
    print(f"  Advantage: {drs_advantage:.3f}s, Usage probability: {drs_usage:.2f} (n={drs_samples})")
    
else:
    print(f"\nUSING FALLBACK PARAMETERS:")
    print("- Run F1_Parameter_Extractor.py 'Dutch Grand Prix' for data-driven parameters")
    print("- Current tire/position/error parameters are estimates")

print(f"\nWET WEATHER STRATEGY CHANGES:")
print("- Removed WET tire compound (never used in practice)")
print("- Wet strategies now use only INTERMEDIATE compound")
print("- Updated wet strategy profiles for realistic tire usage")

print("\n" + "="*80)