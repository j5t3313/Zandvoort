#!/usr/bin/env python3
"""
Dutch Grand Prix Tire Stint Strategy Analyzer
Extracts and analyzes tire strategies for drivers in the Dutch GP from the last 5 seasons.
"""

import fastf1
import pandas as pd
from collections import defaultdict
import warnings

# Suppress FastF1 warnings for cleaner output
warnings.filterwarnings('ignore')


def analyze_tire_stints(session):
    """
    Analyze tire stints for all drivers in a session.
    
    Args:
        session: FastF1 session object
        
    Returns:
        dict: Driver tire stint data
    """
    # Get laps data
    laps = session.laps
    
    # Dictionary to store stint data for each driver
    driver_stints = defaultdict(list)
    
    # Get unique drivers
    drivers = laps['Driver'].unique()
    
    for driver in drivers:
        if pd.isna(driver):
            continue
            
        driver_laps = laps[laps['Driver'] == driver].copy()
        driver_laps = driver_laps.sort_values('LapNumber')
        
        if len(driver_laps) == 0:
            continue
            
        current_compound = None
        stint_start_lap = None
        stint_laps = 0
        
        for _, lap in driver_laps.iterrows():
            lap_compound = lap['Compound']
            lap_number = lap['LapNumber']
            
            # Skip if compound data is missing
            if pd.isna(lap_compound):
                continue
                
            # If this is a new compound or first lap
            if current_compound != lap_compound:
                # Save previous stint if it exists
                if current_compound is not None and stint_laps > 0:
                    driver_stints[driver].append({
                        'compound': current_compound,
                        'start_lap': stint_start_lap,
                        'laps_completed': stint_laps
                    })
                
                # Start new stint
                current_compound = lap_compound
                stint_start_lap = lap_number
                stint_laps = 1
            else:
                stint_laps += 1
        
        # Don't forget the last stint
        if current_compound is not None and stint_laps > 0:
            driver_stints[driver].append({
                'compound': current_compound,
                'start_lap': stint_start_lap,
                'laps_completed': stint_laps
            })
    
    return dict(driver_stints)

def format_driver_strategy(driver_code, stints):
    """
    Format a driver's tire strategy for display.
    
    Args:
        driver_code: Three-letter driver code
        stints: List of stint dictionaries
        
    Returns:
        str: Formatted strategy string
    """
    if not stints:
        return f"{driver_code}: No data"
    
    strategy_parts = []
    for stint in stints:
        compound = stint['compound']
        laps = stint['laps_completed']
        strategy_parts.append(f"{compound}({laps})")
    
    strategy = " → ".join(strategy_parts)
    return f"{driver_code}: {strategy}"

def analyze_dutch_gp_seasons():
    """
    Analyze tire strategies for Dutch GP from the last 4 seasons.
    """
    # Dutch GP returned in 2021 after a long absence
    seasons = [2021, 2022, 2023, 2024]
    
    print("Dutch Grand Prix Tire Stint Analysis")
    print("=" * 50)
    
    for year in seasons:
        try:
            print(f"\n🏁 {year} Dutch Grand Prix")
            print("-" * 30)
            
            # Load the session
            session = fastf1.get_session(year, 'Netherlands', 'R')
            session.load()
            
            # Analyze tire stints
            driver_stints = analyze_tire_stints(session)
            
            if not driver_stints:
                print("❌ No tire data available for this race")
                continue
            
            # Sort drivers by final position (or alphabetically if position not available)
            try:
                results = session.results
                if not results.empty:
                    # Sort by position
                    sorted_drivers = results.sort_values('Position')['Abbreviation'].tolist()
                    # Filter to only drivers we have stint data for
                    sorted_drivers = [d for d in sorted_drivers if d in driver_stints]
                else:
                    sorted_drivers = sorted(driver_stints.keys())
            except:
                sorted_drivers = sorted(driver_stints.keys())
            
            # Display strategies
            for driver in sorted_drivers:
                stints = driver_stints[driver]
                print(format_driver_strategy(driver, stints))
            
            # Summary statistics
            total_drivers = len(driver_stints)
            print(f"\n📊 Summary: {total_drivers} drivers analyzed")
            
            # Compound usage summary
            compound_usage = defaultdict(int)
            for stints in driver_stints.values():
                for stint in stints:
                    compound_usage[stint['compound']] += 1
            
            if compound_usage:
                print("Tire compound usage:")
                for compound, count in sorted(compound_usage.items()):
                    print(f"  • {compound}: {count} stints")
                    
        except Exception as e:
            print(f"❌ Error analyzing {year}: {str(e)}")
            if "No data available" in str(e):
                print("   (Race may not have occurred yet)")
            continue

def main():
    """Main function to run the analysis."""
    try:
        analyze_dutch_gp_seasons()
        
        print("\n" + "=" * 50)
        print("Analysis complete!")
        print("\nLegend:")
        print("• SOFT(15) = 15 laps on soft compound")
        print("• MEDIUM(20) = 20 laps on medium compound") 
        print("• HARD(25) = 25 laps on hard compound")
        print("• Strategy shows: Compound1 → Compound2 → Compound3...")
        
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        print("Make sure you have fastf1 installed: pip install fastf1")

if __name__ == "__main__":
    main()