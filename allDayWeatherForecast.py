import requests
from datetime import datetime

def get_daily_forecast(date_string, lat=52.3888, lon=4.5409, api_key="APIKEY"):
    """
    Get all weather forecasts for a specific date.
    
    Args:
        date_string (str): Date in format "YYYY-MM-DD" (e.g., "2025-08-31")
        lat (float): Latitude (default: Zandvoort Circuit)
        lon (float): Longitude (default: Zandvoort Circuit)
        api_key (str): OpenWeatherMap API key
    
    Returns:
        list: List of forecast dictionaries for the specified date
    """
    
    # Build API URL
    weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(weather_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        weather_data = response.json()
        
        # Filter forecasts for the specified date
        daily_forecasts = []
        target_date = date_string
        
        for forecast in weather_data["list"]:
            forecast_date = forecast["dt_txt"].split(" ")[0]  # Extract date part
            
            if forecast_date == target_date:
                forecast_info = {
                    "time": forecast["dt_txt"],
                    "temperature": round(forecast["main"]["temp"], 1),
                    "feels_like": round(forecast["main"]["feels_like"], 1),
                    "rain_probability": round(forecast["pop"] * 100, 1),
                    "humidity": forecast["main"]["humidity"],
                    "clouds": forecast["clouds"]["all"],
                    "weather": forecast["weather"][0]["description"],
                    "wind_speed": round(forecast["wind"]["speed"], 1),
                    "rain_volume": forecast.get("rain", {}).get("3h", 0)  # 3-hour rain volume
                }
                daily_forecasts.append(forecast_info)
        
        return daily_forecasts
        
    except requests.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return []
    except KeyError as e:
        print(f"Unexpected API response format: {e}")
        return []

def print_daily_forecast(date_string, lat=52.3888, lon=4.5409, api_key="APIKEY"):
    """
    Print formatted daily forecast for a specific date.
    """
    forecasts = get_daily_forecast(date_string, lat, lon, api_key)
    
    if not forecasts:
        print(f"No forecast data found for {date_string}")
        return
    
    print(f"\n🌤️  Weather Forecast for {date_string}")
    print(f"📍 Location: {lat:.4f}, {lon:.4f} (Zandvoort Circuit)")
    print("=" * 80)
    
    for forecast in forecasts:
        time_part = forecast["time"].split(" ")[1]  # Extract time part
        
        print(f"⏰ {time_part}")
        print(f"   🌡️  Temperature: {forecast['temperature']}°C (feels like {forecast['feels_like']}°C)")
        print(f"   🌧️  Rain Probability: {forecast['rain_probability']}%")
        
        if forecast['rain_volume'] > 0:
            print(f"   ☔ Rain Volume (3h): {forecast['rain_volume']}mm")
        
        print(f"   ☁️  Weather: {forecast['weather'].title()}")
        print(f"   💨 Wind Speed: {forecast['wind_speed']} m/s")
        print(f"   💧 Humidity: {forecast['humidity']}%")
        print(f"   ☁️  Cloud Cover: {forecast['clouds']}%")
        print("-" * 40)
    
    # Summary
    max_rain_prob = max(f['rain_probability'] for f in forecasts)
    avg_temp = sum(f['temperature'] for f in forecasts) / len(forecasts)
    
    print(f"\n📊 Daily Summary:")
    print(f"   Highest Rain Probability: {max_rain_prob}%")
    print(f"   Average Temperature: {avg_temp:.1f}°C")
    print(f"   Total Forecast Points: {len(forecasts)}")

# Example usage
if __name__ == "__main__":
    API_KEY = "APIKEY"  # Replace with your actual API key
    
    # Get forecast for August 31, 2025 (race day)
    race_date = "2025-08-31"
    
    print_daily_forecast(race_date, api_key=API_KEY)
    
    # Alternative: Get raw data for further processing
    # forecasts = get_daily_forecast(race_date, api_key=API_KEY)
    # for forecast in forecasts:
    #     print(f"{forecast['time']}: {forecast['rain_probability']}% rain, {forecast['temperature']}°C")