# Zandvoort Strategy Analysis

A Formula 1 race strategy simulation system that uses historical data extraction, Bayesian modeling, and Monte Carlo simulation to predict optimal tire strategies for the Dutch Grand Prix at Zandvoort.

## 🏁 Overview

This project combines data science with F1 domain knowledge to create a strategy simulation tool. Using historical FastF1 data, the system extracts track-specific parameters and simulates thousands of race scenarios to determine optimal tire strategies for different grid positions.

## ✨ Key Features

- **Data-Driven Parameter Extraction**: Automatically extracts tire performance, position penalties, driver error rates, and DRS effectiveness from historical race data
- **Bayesian Tire Modeling**: Uses MCMC simulation with JAX/NumPyro for sophisticated tire degradation modeling
- **Monte Carlo Race Simulation**: Runs hundreds of race simulations incorporating weather, safety cars, and strategic variables
- **Comprehensive Validation Framework**: Validates predictions against actual race results with statistical rigor
- **Track-Specific Optimization**: Zandvoort-specific parameters including overtaking difficulty, weather patterns, and circuit characteristics

## 📊 Example Output

```
GRID POSITION 5
--------------------------------------------------
Strategy                 | Avg Final Pos | Avg Points | Top 5 % | Podium %
1-stop (M-H)            | 4.2           | 10.8       | 78%     | 23%
2-stop (S-M-H)          | 4.6           | 9.4        | 72%     | 18%
2-stop (S-H-M)          | 5.1           | 8.2        | 64%     | 12%
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install fastf1 pandas numpy matplotlib seaborn scipy scikit-learn
pip install jax numpyro tqdm
```

### Basic Usage

1. **Extract Parameters** (Run first for best accuracy):
```bash
python F1_Parameter_Extractor.py "Dutch Grand Prix"
```

2. **Run Strategy Simulation**:
```bash
python stochasticPitSimv3_Dutch.py
```

3. **Validate Against Race Results** (after race occurs):
```bash
python dutchStrategyModelValidation.py
```

4. **Analyze Historical Strategies**:
```bash
python zandvoortHistoricalStrategies.py
```

## 📁 Project Structure

```
├── F1_Parameter_Extractor.py          # Historical data extraction and parameter generation
├── stochasticPitSimv3_Dutch.py        # Main simulation engine with Zandvoort parameters
├── dutchStrategyModelValidation.py    # Validation framework for race results
├── zandvoortHistoricalStrategies.py   # Historical strategy analysis
└── dutch_grand_prix_targeted_parameters.py  # Generated parameter file (created by extractor)
```

## 🔧 Core Components

### 1. Parameter Extraction (`F1_Parameter_Extractor.py`)
- Extracts tire performance curves using SOFT as baseline
- Calculates position-based penalties for traffic/dirty air
- Determines driver error rates from lap time anomalies
- Estimates DRS effectiveness from pace improvements
- Uses 3+ years of historical data with statistical filtering

### 2. Race Simulation (`stochasticPitSimv3_Dutch.py`)
- **Tire Modeling**: Bayesian MCMC degradation with compound-specific parameters
- **Weather Simulation**: Realistic rain patterns (probability based on Open Weather API forecast)
- **Safety Car Events**: 67% SC/VSC probability based on Formula 1 official information
- **Strategy Implementation**: 9 different tire strategies 
- **Position Effects**: Traffic penalties, DRS usage, overtaking difficulty

### 3. Validation Framework (`dutchStrategyModelValidation.py`)
- Loads actual race results from FastF1
- Compares predictions vs. reality with statistical metrics (MAE, RMSE)
- Validates weather predictions and safety car occurrences
- Generates comprehensive accuracy reports and visualizations

## 📈 Zandvoort-Specific Parameters

The simulation is tuned for Circuit Zandvoort characteristics:

| Parameter | Value | Source |
|-----------|-------|---------|
| Base Lap Time | 71.0s | Historical data |
| Race Distance | 72 laps | Official |
| Pit Time Loss | 21.52s | Historical data |
| Rain Probability | 30% | Coastal climate |
| Safety Car Rate | 67% | Historical data |
| Overtaking Difficulty | High | Banking/narrow track |

## 🌧️ Weather & Strategy Modeling

### Weather Scenarios
- **Early Shower**: Laps 5-15
- **Mid-Race Rain**: Laps 20-35  
- **Late Drama**: Laps 40+
- **Brief Shower**: 3-6 lap duration

### Tire Strategies
- **Dry Strategies**: 1-stop (M-H), 2-stop (S-M-H), aggressive 2-stop variants
- **Wet Strategies**: Conservative and aggressive intermediate strategies
- **Mixed Conditions**: Gamble on dry, weather-reactive strategies

*Note: WET compound removed from model as it's never used in practice*

## 🎯 Validation Results

The system includes comprehensive validation against actual race results:

- **Position Prediction Accuracy**: Tracks mean absolute error by grid position
- **Weather Validation**: Compares predicted vs. actual rain occurrence
- **Strategy Effectiveness**: Validates strategy success rates
- **Parameter Quality Assessment**: Evaluates extraction reliability

## 📊 Visualizations

The system generates professional-grade analysis plots:
- Race time distribution comparisons
- Final position probability distributions
- Strategy effectiveness heatmaps by grid position
- Validation accuracy metrics

## 🔬 Technical Approach

### Statistical Methods
- **Bayesian Inference**: MCMC sampling for tire degradation
- **Monte Carlo Simulation**: 300+ iterations per scenario
- **Outlier Filtering**: 2.5σ statistical filtering
- **Uncertainty Quantification**: Confidence intervals and error bounds

### F1 Domain Modeling
- **Non-linear Tire Degradation**: Exponential degradation for long stints
- **Track Evolution**: Improving lap times throughout the race
- **Fuel Effects**: -0.035s/kg standard F1 impact
- **Temperature Effects**: Compound-specific optimal windows

## ⚠️ Limitations

This system maximizes accuracy within **publicly available data constraints**:

- No real-time telemetry (tire temps, fuel flow, brake temps)
- No team strategy intelligence or radio communications  
- No car setup data affecting tire performance
- No precise aerodynamic wake modeling

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional circuit implementations
- Machine learning integration
- Advanced aerodynamic modeling

## 📚 Data Sources

- **FastF1 API**: Historical lap times, tire data, race results
- **Official F1 Data**: Pit lane measurements, fuel regulations
- **Weather Patterns**: Historical Zandvoort climate data
- **Weather Forecast**: Open Weather API
- **Circuit Analysis**: Track characteristics and safety statistics as published by Formula 1

## 📜 License

This project is for educational and research purposes. Formula 1 is a trademark of Formula One Licensing BV. For other licensing information, see License document.

## 🙏 Acknowledgments

- **FastF1**: For providing comprehensive F1 data access
- **Open Weather**: For providing weather forecast information
- **JAX/NumPyro**: For Bayesian modeling capabilities
- **F1 Community**: For domain knowledge and validation data

---
