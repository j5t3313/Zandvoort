"""
Extracted F1 simulation parameters for Dutch Grand Prix
Targeted extraction: Position penalties, tire performance, driver errors, DRS
Generated automatically from historical FastF1 data
"""

import numpy as np

# POSITION PENALTIES
POSITION_PENALTIES = {1: {'penalty': 5.677999999999997, 'std': 2.419959273986882, 'sample_size': 7}, 2: {'penalty': 5.587999999999994, 'std': 0.3148115397433134, 'sample_size': 6}, 3: {'penalty': 5.8870000000000005, 'std': 0.32358281783803206, 'sample_size': 5}, 4: {'penalty': 5.8075000000000045, 'std': 0.5299244180908163, 'sample_size': 6}, 5: {'penalty': 5.665000000000006, 'std': 0.7155585510634324, 'sample_size': 5}, 6: {'penalty': 5.300999999999988, 'std': 0.8177018772144291, 'sample_size': 5}, 7: {'penalty': 6.061999999999998, 'std': 2.656431189898708, 'sample_size': 6}, 8: {'penalty': 6.046500000000002, 'std': 0.6100487685423173, 'sample_size': 4}, 9: {'penalty': 5.4594999999999985, 'std': 2.9812040317909565, 'sample_size': 6}, 10: {'penalty': 5.336999999999989, 'std': 1.193647107970924, 'sample_size': 7}, 11: {'penalty': 5.458000000000013, 'std': 3.9138877443675124, 'sample_size': 7}, 12: {'penalty': 5.361000000000004, 'std': 3.500352767936398, 'sample_size': 7}, 13: {'penalty': 5.384749999999997, 'std': 3.3828443653233538, 'sample_size': 6}, 14: {'penalty': 5.453000000000003, 'std': 3.60831758025815, 'sample_size': 5}, 15: {'penalty': 4.969000000000008, 'std': 4.073700695927475, 'sample_size': 5}, 16: {'penalty': 5.227999999999994, 'std': 4.39111978884657, 'sample_size': 4}, 17: {'penalty': 5.314999999999991, 'std': 4.078894393364726, 'sample_size': 6}, 18: {'penalty': 4.373000000000005, 'std': 3.8362490273703505, 'sample_size': 5}, 19: {'penalty': 0.6324999999999932, 'std': 5.126221342275421, 'sample_size': 4}, 20: {'penalty': 4.14650000000001, 'std': 5.128012209640519, 'sample_size': 6}}

# TIRE PERFORMANCE
TIRE_PERFORMANCE = {'SOFT': {'base_time': 76.64084856439149, 'degradation_rate': 0, 'r_squared': 0.0021090653310387175, 'sample_size': 1315, 'offset': 0.0}, 'MEDIUM': {'base_time': 76.98092501067626, 'degradation_rate': 0, 'r_squared': 0.005392836814875457, 'sample_size': 1321, 'offset': 0.34007644628476896}, 'HARD': {'base_time': 76.24924875248799, 'degradation_rate': 0, 'r_squared': 0.0017685164116332253, 'sample_size': 994, 'offset': -0.3915998119035038}}

# DRIVER ERROR RATES
DRIVER_ERROR_RATES = {'dry': {'base_error_rate': 0.046536796536796536, 'mean_error_rate': 0.049738012595643484, 'std_error_rate': 0.03399333405495651, 'sample_size': 202}, 'wet': {'base_error_rate': 0.0, 'mean_error_rate': 0.005952380952380952, 'std_error_rate': 0.0266198568749975, 'sample_size': 21}}

# DRS EFFECTIVENESS
DRS_EFFECTIVENESS = {'mean_advantage': 0.33933898305084786, 'median_advantage': 0.28700000000000614, 'std_advantage': 0.19186553909050796, 'sample_size': 1062, 'usage_probability': 0.3}

# CONVENIENCE FUNCTIONS

def get_position_penalty(position):
    """Get traffic/dirty air penalty for grid position"""
    if position in POSITION_PENALTIES:
        return POSITION_PENALTIES[position]["penalty"]
    else:
        # Extrapolate for positions beyond data
        if position <= 20:
            return 0.05 * (position - 1)  # Linear approximation
        else:
            return 1.0  # High penalty for back of grid

def get_tire_offset(compound):
    """Get tire compound offset relative to SOFT"""
    return TIRE_PERFORMANCE.get(compound, {}).get("offset", 0.0)

def get_tire_degradation_rate(compound):
    """Get tire degradation rate in s/lap"""
    return TIRE_PERFORMANCE.get(compound, {}).get("degradation_rate", 0.08)

def get_driver_error_rate(weather_condition="dry"):
    """Get driver error probability per lap"""
    return DRIVER_ERROR_RATES.get(weather_condition, {}).get("base_error_rate", 0.01)

def get_drs_advantage():
    """Get DRS time advantage in seconds"""
    mean_adv = DRS_EFFECTIVENESS.get("median_advantage", 0.25)
    std_adv = DRS_EFFECTIVENESS.get("std_advantage", 0.1)
    return max(0.1, np.random.normal(mean_adv, std_adv))

def get_drs_usage_probability():
    """Get probability of being in DRS range"""
    return DRS_EFFECTIVENESS.get("usage_probability", 0.3)

