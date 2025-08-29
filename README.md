# Zandvoort Strategy Analysis

A Formula 1 race strategy simulation system that uses historical data extraction, practice-based tire modeling, Bayesian inference, and Monte Carlo simulation to predict optimal tire strategies for the Dutch Grand Prix at Zandvoort.

## Overview

This project combines data science with F1 domain knowledge to create a strategy simulation tool. Using historical FastF1 data and real-time practice session analysis, the system extracts track-specific parameters and simulates thousands of race scenarios to determine optimal tire strategies for different grid positions.

## Key Features

- **Data-Driven Parameter Extraction**: Automatically extracts tire performance, position penalties, driver error rates, and DRS effectiveness from historical race data (2021 - 2024)
- **Practice-Based Tire Modeling**: Uses FP1+FP2 session data with Bayesian MCMC to create accurate tire degradation models
- **Hierarchical Modeling Approach**: Three-tier fallback system (practice models → extracted parameters → defaults) ensures reliability
- **Monte Carlo Race Simulation**: Runs hundreds of race simulations incorporating weather, safety cars, and strategic variables
- **Comprehensive Validation Framework**: Validates predictions against actual race results with statistical rigor
- **Track-Specific Optimization**: Zandvoort-specific parameters including overtaking difficulty, weather patterns, and circuit characteristics

## Example Output

```
GRID POSITION 5
--------------------------------------------------
Strategy                 | Avg Final Pos | Avg Points | Top 5 % | Podium %
1-stop (M-H)            | 4.2           | 10.8       | 78%     | 23%
2-stop (S-M-H)          | 4.6           | 9.4        | 72%     | 18%
2-stop (S-H-M)          | 5.1           | 8.2        | 64%     | 12%
```

## Quick Start

### Prerequisites

```bash
pip install fastf1 pandas numpy matplotlib seaborn scipy scikit-learn
pip install jax numpyro tqdm
```

### Basic Usage

**Pre-Race Setup (Run before race weekend):**

1. **Extract Historical Parameters**:
```bash
python F1_Parameter_Extractor.py "Dutch Grand Prix"
```

**Race Weekend Workflow:**

2. **Build Practice-Based Tire Models** (after FP1/FP2 sessions):
```bash
python fp1_fp2_tire_model.py
```

3. **Run Strategy Simulation** (after qualifying):
```bash
python stochasticPitSimv4_Dutch.py
```

**Post-Race Analysis:**

4. **Validate Against Race Results** (after race completion):
```bash
python dutchStrategyModelValidation.py
```

## Project Structure

```
├── F1_Parameter_Extractor.py          # Historical data extraction and parameter generation
├── stochasticPitSimv4_Dutch.py        # Main simulation engine with Zandvoort parameters
├── fp1_fp2_tire_model.py              # Practice-based tire modeling system
├── dutchStrategyModelValidation.py    # Validation framework for race results
└── dutch_grand_prix_targeted_parameters.py  # Generated parameter file (created by extractor)
```

## Core Components

### 1. Parameter Extraction (`F1_Parameter_Extractor.py`)
- Extracts tire performance curves using SOFT as baseline
- Calculates position-based penalties for traffic/dirty air
- Determines driver error rates from lap time anomalies
- Estimates DRS effectiveness from pace improvements
- Uses 3+ years of historical data with statistical filtering

### 2. Practice-Based Tire Modeling (`fp1_fp2_tire_model.py`)
- **Dual Session Integration**: Combines FP1 and FP2 data for comprehensive tire analysis
- **Bayesian MCMC Modeling**: Uses NUTS sampling with compound-specific priors
- **Quality Assessment**: Uncertainty quantification and confidence scoring per compound
- **Session-Aware Processing**: Tracks data sources and handles single-session scenarios
- **Fallbacks**: Hierarchical degradation to extracted parameters if practice models fail

### 3. Race Simulation (`stochasticPitSimv4_Dutch.py`)
- **Enhanced Tire Modeling**: Primary use of practice-based models with intelligent fallbacks
- **Weather Simulation**: Realistic rain patterns (probability based on Open Weather API forecast)
- **Safety Car Events**: 67% SC/VSC probability based on Formula 1 official information
- **Strategy Implementation**: 9 different tire strategies optimized for Zandvoort based on historical usage
- **Position Effects**: Traffic penalties, DRS usage, overtaking difficulty

### 4. Validation Framework (`dutchStrategyModelValidation.py`)
- Loads actual race results from FastF1
- Compares predictions vs. reality with statistical metrics (MAE, RMSE)
- Validates weather predictions and safety car occurrences
- Assesses practice model accuracy against race degradation patterns
- Generates comprehensive accuracy reports and visualizations

## Zandvoort-Specific Parameters

The simulation is tuned for Circuit Zandvoort characteristics:

| Parameter | Value | Source |
|-----------|-------|---------|
| Base Lap Time | 71.0s | Historical data |
| Race Distance | 72 laps | Official |
| Pit Time Loss | 16.5s | Approximation based on historical data + new pitlane speed limit for 2025 |
| Rain Probability | 30% | Open Weather API |
| Safety Car Rate | 67% | Historical data |
| Overtaking Difficulty | High | Banking/narrow track |

## Enhanced Tire Modeling System

### Three-Tier Modeling Hierarchy

1. **Practice-Based Models (Primary)**
   - FP1+FP2 Bayesian MCMC tire curves
   - Track-specific degradation from actual sessions
   - Compound-specific uncertainty quantification

2. **Extracted Historical Parameters (Secondary)**
   - Multi-year statistical analysis
   - Race-validated tire performance
   - Position penalties and driver error rates

3. **Fallback Defaults (Tertiary)**
   - Industry-standard assumptions
   - Ensures simulation reliability
   - Conservative parameter estimates

### Model Quality Indicators

- **Excellent**: Both FP1+FP2 available, R² > 0.8, >50 practice laps
- **Good**: Single session or R² > 0.6, 20-50 practice laps
- **Moderate**: Limited data, falls back to extracted parameters

## Weather & Strategy Modeling

### Weather Scenarios
- **Early Shower**: Laps 5-15
- **Mid-Race Rain**: Laps 20-35  
- **Late Drama**: Laps 40+
- **Brief Shower**: 3-6 lap duration

### Tire Strategies
- **Dry Strategies**: 1-stop, 2-stop, aggressive 2-stop variants
- **Wet Strategies**: Conservative and aggressive intermediate strategies
- **Mixed Conditions**: Gamble on dry, weather-reactive strategies

*Note: WET compound removed from model as it's never used in practice*

## Validation Results

The system includes comprehensive validation against actual race results:

- **Position Prediction Accuracy**: Tracks mean absolute error by grid position
- **Weather Validation**: Compares predicted vs. actual rain occurrence
- **Strategy Effectiveness**: Validates strategy success rates
- **Practice Model Assessment**: Evaluates FP1+FP2 tire model accuracy
- **Parameter Quality Evaluation**: Multi-tier data source reliability analysis

## Visualizations

The system generates analysis plots:
- Race time distribution comparisons
- Final position probability distributions
- Strategy effectiveness heatmaps by grid position
- Practice-based tire degradation curves
- Model confidence and data quality indicators
- Validation accuracy metrics

## Technical Approach

### Statistical Methods
- **Bayesian Inference**: MCMC sampling for tire degradation with compound-specific priors
- **Monte Carlo Simulation**: 300+ iterations per scenario
- **Outlier Filtering**: 2.5σ statistical filtering across practice sessions
- **Uncertainty Quantification**: Confidence intervals and error bounds
- **Multi-Session Integration**: Intelligent combination of FP1+FP2 data

### F1 Domain Modeling
- **Practice-Based Degradation**: Real track surface characteristics from session data
- **Non-linear Tire Degradation**: Exponential degradation for long stints
- **Track Evolution**: Session-specific track improvement modeling
- **Fuel Effects**: -0.035s/kg standard F1 impact
- **Temperature Effects**: Compound-specific optimal windows derived from practice

## Enhanced Accuracy Features

### Practice Session Integration
- **Morning/Afternoon Conditions**: FP1 and FP2 capture different track states
- **Weather Resilience**: If one session is wet, dry models still available
- **Increased Sample Size**: Combined sessions provide more reliable statistics
- **Session-Specific Analysis**: Tracks which session contributed to each parameter

### Fallback System
- **Intelligent Degradation**: Practice models → Historical extraction → Defaults
- **Quality Transparency**: Clear indication of data source reliability
- **Graceful Handling**: System remains functional even with limited data
- **Parameter Provenance**: Tracks origin of each simulation parameter

## Limitations

This system maximizes accuracy within **publicly available data constraints**:

- **Practice vs Race Gap**: FP1/FP2 conditions may not represent full race stress
- **Limited Stint Lengths**: Practice stints typically shorter than race stints
- **No Real-Time Telemetry**: No access to tire temps, fuel flow, brake temps
- **No Team Intelligence**: No access to strategy communications or setup data
- **Setup Dependencies**: Practice setup may differ from race configuration
- **Weather Transition Modeling**: Binary dry/wet states vs gradual changes

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional circuit implementations
- Machine learning integration for practice-race correlation
- Advanced aerodynamic modeling
- Real-time weather integration

## Data Sources

- **FastF1 API**: Historical lap times, tire data, race results, practice sessions
- **Official F1 Data**: Pit lane measurements, fuel regulations
- **Weather Patterns**: Historical Zandvoort climate data
- **Weather Forecast**: Open Weather API
- **Circuit Analysis**: Track characteristics and safety statistics as published by Formula 1

## License

This project is for educational and research purposes. Formula 1 is a trademark of Formula One Licensing BV. For other licensing information, see License document.

## Acknowledgments

- **FastF1**: For providing comprehensive F1 data access including practice sessions
- **Open Weather**: For providing weather forecast information
- **JAX/NumPyro**: For Bayesian modeling capabilities
- **F1 Community**: For domain knowledge and validation data

---
