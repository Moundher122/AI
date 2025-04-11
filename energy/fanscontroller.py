import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import datetime
import time

class IntelligentHVACSystem:
    def __init__(self):
        # Input variables - define universes
        self.current_temp = ctrl.Antecedent(np.arange(10, 40, 0.1), 'current_temp')
        self.humidity = ctrl.Antecedent(np.arange(0, 100, 1), 'humidity')
        self.forecast_temp = ctrl.Antecedent(np.arange(10, 40, 0.1), 'forecast_temp')
        self.outdoor_temp = ctrl.Antecedent(np.arange(0, 45, 0.1), 'outdoor_temp')
        self.outdoor_humidity = ctrl.Antecedent(np.arange(0, 100, 1), 'outdoor_humidity')
        self.temp_stability = ctrl.Antecedent(np.arange(0, 101, 1), 'temp_stability')
        
        # Output variables
        self.fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')
        self.ac_control = ctrl.Consequent(np.arange(0, 101, 1), 'ac_control')
        self.energy_mode = ctrl.Consequent(np.arange(0, 101, 1), 'energy_mode')
        
        # New air exchange control variables
        self.intake_fan = ctrl.Consequent(np.arange(0, 101, 1), 'intake_fan')  # Brings outside air in
        self.exhaust_fan = ctrl.Consequent(np.arange(0, 101, 1), 'exhaust_fan')  # Pushes inside air out
        
        # Set up membership functions for inputs
        # Temperature membership functions
        self.current_temp['cold'] = fuzz.trimf(self.current_temp.universe, [10, 15, 20])
        self.current_temp['comfortable'] = fuzz.trimf(self.current_temp.universe, [18, 22, 25])
        self.current_temp['warm'] = fuzz.trimf(self.current_temp.universe, [23, 27, 30])
        self.current_temp['hot'] = fuzz.trimf(self.current_temp.universe, [28, 35, 40])
        
        # Humidity membership functions
        self.humidity['dry'] = fuzz.trimf(self.humidity.universe, [0, 15, 30])
        self.humidity['normal'] = fuzz.trimf(self.humidity.universe, [25, 45, 65])
        self.humidity['humid'] = fuzz.trimf(self.humidity.universe, [60, 80, 100])
        
        # Forecast temperature membership functions
        self.forecast_temp['cooling'] = fuzz.trimf(self.forecast_temp.universe, [10, 15, 22])
        self.forecast_temp['stable'] = fuzz.trimf(self.forecast_temp.universe, [18, 25, 32])
        self.forecast_temp['warming'] = fuzz.trimf(self.forecast_temp.universe, [28, 35, 40])
        
        # Outdoor temperature membership functions
        self.outdoor_temp['cold'] = fuzz.trimf(self.outdoor_temp.universe, [0, 10, 18])
        self.outdoor_temp['pleasant'] = fuzz.trimf(self.outdoor_temp.universe, [16, 22, 28])
        self.outdoor_temp['hot'] = fuzz.trimf(self.outdoor_temp.universe, [26, 35, 45])
        
        # Outdoor humidity membership functions
        self.outdoor_humidity['dry'] = fuzz.trimf(self.outdoor_humidity.universe, [0, 20, 40])
        self.outdoor_humidity['moderate'] = fuzz.trimf(self.outdoor_humidity.universe, [30, 50, 70])
        self.outdoor_humidity['humid'] = fuzz.trimf(self.outdoor_humidity.universe, [60, 80, 100])
        
        # Temperature stability membership functions
        self.temp_stability['unstable'] = fuzz.trimf(self.temp_stability.universe, [0, 0, 40])
        self.temp_stability['stabilizing'] = fuzz.trimf(self.temp_stability.universe, [30, 50, 70])
        self.temp_stability['stable'] = fuzz.trimf(self.temp_stability.universe, [60, 100, 100])
        
        # Set up membership functions for outputs
        # Fan speed membership functions
        self.fan_speed['off'] = fuzz.trimf(self.fan_speed.universe, [0, 0, 10])
        self.fan_speed['low'] = fuzz.trimf(self.fan_speed.universe, [5, 25, 45])
        self.fan_speed['medium'] = fuzz.trimf(self.fan_speed.universe, [40, 60, 80])
        self.fan_speed['high'] = fuzz.trimf(self.fan_speed.universe, [75, 100, 100])
        
        # AC control membership functions
        self.ac_control['off'] = fuzz.trimf(self.ac_control.universe, [0, 0, 40])
        self.ac_control['on'] = fuzz.trimf(self.ac_control.universe, [60, 100, 100])
        
        # Energy mode membership functions
        self.energy_mode['eco'] = fuzz.trimf(self.energy_mode.universe, [0, 0, 40])
        self.energy_mode['balanced'] = fuzz.trimf(self.energy_mode.universe, [30, 50, 70])
        self.energy_mode['performance'] = fuzz.trimf(self.energy_mode.universe, [60, 100, 100])
        
        # Air exchange fan membership functions
        self.intake_fan['off'] = fuzz.trimf(self.intake_fan.universe, [0, 0, 10])
        self.intake_fan['low'] = fuzz.trimf(self.intake_fan.universe, [5, 25, 45])
        self.intake_fan['medium'] = fuzz.trimf(self.intake_fan.universe, [40, 60, 80])
        self.intake_fan['high'] = fuzz.trimf(self.intake_fan.universe, [75, 100, 100])
        
        self.exhaust_fan['off'] = fuzz.trimf(self.exhaust_fan.universe, [0, 0, 10])
        self.exhaust_fan['low'] = fuzz.trimf(self.exhaust_fan.universe, [5, 25, 45])
        self.exhaust_fan['medium'] = fuzz.trimf(self.exhaust_fan.universe, [40, 60, 80])
        self.exhaust_fan['high'] = fuzz.trimf(self.exhaust_fan.universe, [75, 100, 100])
        
        # Define fuzzy rules
        self.define_rules()
        
        # Create control system
        self.hvac_ctrl = ctrl.ControlSystem(self.rules)
        self.hvac_simulation = ctrl.ControlSystemSimulation(self.hvac_ctrl)
        
        # Initialize historical data storage
        self.history = {
            'timestamps': [],
            'current_temp': [],
            'humidity': [],
            'outdoor_temp': [],
            'outdoor_humidity': [],
            'forecast_temp': [],
            'fan_speed': [],
            'ac_control': [],
            'intake_fan': [],
            'exhaust_fan': [],
            'energy_mode': [],
            'temp_stability': [],
            'energy_saved': []
        }
        
        # Temperature tracking for stability calculation
        self.temp_history = []
        self.target_temp = 22  # Default target temperature
        self.stability_window = 10  # Number of readings to evaluate stability
        
        # Energy consumption baseline (kWh)
        self.ac_power = 2.5  # AC power consumption in kW
        self.fan_power = 0.2  # Internal fan power consumption in kW
        self.intake_power = 0.15  # Intake fan power consumption in kW  
        self.exhaust_power = 0.15  # Exhaust fan power consumption in kW
        
    def define_rules(self):
        self.rules = [
            # Natural ventilation rules - use outside air when beneficial
            # When outside is pleasant and inside is warm/hot, use outside air
            ctrl.Rule(self.outdoor_temp['pleasant'] & self.current_temp['warm'], self.intake_fan['high']),
            ctrl.Rule(self.outdoor_temp['pleasant'] & self.current_temp['warm'], self.exhaust_fan['high']),
            ctrl.Rule(self.outdoor_temp['pleasant'] & self.current_temp['hot'], self.intake_fan['high']),
            ctrl.Rule(self.outdoor_temp['pleasant'] & self.current_temp['hot'], self.exhaust_fan['high']),
            
            # When outside is cold and inside is hot, use outside air
            ctrl.Rule(self.outdoor_temp['cold'] & self.current_temp['hot'], self.intake_fan['medium']),
            ctrl.Rule(self.outdoor_temp['cold'] & self.current_temp['hot'], self.exhaust_fan['medium']),
            
            # When outside is cold and inside is warm, use some outside air
            ctrl.Rule(self.outdoor_temp['cold'] & self.current_temp['warm'], self.intake_fan['low']),
            ctrl.Rule(self.outdoor_temp['cold'] & self.current_temp['warm'], self.exhaust_fan['low']),
            
            # Don't bring in outside air when it's too hot outside
            ctrl.Rule(self.outdoor_temp['hot'] & self.current_temp['comfortable'], self.intake_fan['off']),
            ctrl.Rule(self.outdoor_temp['hot'] & self.current_temp['cold'], self.intake_fan['off']),
            
            # Don't exhaust inside air when it's comfortable and outside is uncomfortable
            ctrl.Rule(self.current_temp['comfortable'] & self.outdoor_temp['hot'], self.exhaust_fan['off']),
            ctrl.Rule(self.current_temp['comfortable'] & self.outdoor_temp['cold'], self.exhaust_fan['off']),
            
            # Consider humidity when using outside air
            ctrl.Rule(self.outdoor_humidity['humid'] & self.humidity['normal'], self.intake_fan['low']),
            ctrl.Rule(self.outdoor_humidity['humid'] & self.humidity['dry'], self.intake_fan['medium']),
            ctrl.Rule(self.outdoor_humidity['dry'] & self.humidity['humid'], self.intake_fan['high']),
            
            # Forecast-based ventilation rules
            ctrl.Rule(self.forecast_temp['warming'] & self.outdoor_temp['pleasant'], self.intake_fan['high']),
            ctrl.Rule(self.forecast_temp['warming'] & self.outdoor_temp['pleasant'], self.exhaust_fan['high']),
            ctrl.Rule(self.forecast_temp['cooling'] & self.outdoor_temp['cold'], self.intake_fan['off']),
            
            # Energy economy rules based on temperature stability
            ctrl.Rule(self.temp_stability['stable'] & self.current_temp['comfortable'], self.fan_speed['off']),
            ctrl.Rule(self.temp_stability['stable'] & self.current_temp['comfortable'], self.ac_control['off']),
            ctrl.Rule(self.temp_stability['stable'], self.energy_mode['eco']),
            
            # Rules for AC control
            ctrl.Rule(self.current_temp['cold'], self.ac_control['off']),
            ctrl.Rule(self.current_temp['comfortable'] & self.temp_stability['unstable'], self.ac_control['off']),
            ctrl.Rule(self.current_temp['warm'] & self.temp_stability['unstable'] & 
                     ~self.outdoor_temp['pleasant'], self.ac_control['on']),
            ctrl.Rule(self.current_temp['hot'] & ~self.outdoor_temp['cold'], self.ac_control['on']),
            
            # Rules for internal fan speed
            ctrl.Rule(self.current_temp['cold'] & self.temp_stability['unstable'], self.fan_speed['low']),
            ctrl.Rule(self.current_temp['comfortable'] & self.humidity['normal'] & 
                     self.temp_stability['unstable'], self.fan_speed['low']),
            ctrl.Rule(self.current_temp['comfortable'] & self.humidity['humid'] & 
                     self.temp_stability['unstable'], self.fan_speed['medium']),
            ctrl.Rule(self.current_temp['warm'] & self.humidity['dry'] & 
                     self.temp_stability['unstable'], self.fan_speed['medium']),
            ctrl.Rule(self.current_temp['warm'] & self.humidity['normal'] & 
                     self.temp_stability['unstable'], self.fan_speed['medium']),
            ctrl.Rule(self.current_temp['warm'] & self.humidity['humid'] & 
                     self.temp_stability['unstable'], self.fan_speed['high']),
            ctrl.Rule(self.current_temp['hot'] & self.temp_stability['unstable'], self.fan_speed['high']),
            
            # Energy mode rules based on forecast
            ctrl.Rule(self.forecast_temp['cooling'] & self.current_temp['warm'], self.energy_mode['eco']),
            ctrl.Rule(self.forecast_temp['stable'] & self.current_temp['comfortable'], self.energy_mode['eco']),
            ctrl.Rule(self.forecast_temp['warming'] & self.current_temp['comfortable'], self.energy_mode['balanced']),
            ctrl.Rule(self.forecast_temp['warming'] & self.current_temp['warm'] & 
                     self.temp_stability['unstable'], self.energy_mode['performance']),
            
            # Special condition rules
            ctrl.Rule(self.outdoor_temp['hot'] & self.current_temp['warm'] & 
                     self.temp_stability['unstable'], self.fan_speed['high']),
            ctrl.Rule(self.outdoor_temp['cold'] & self.current_temp['cold'] & 
                     self.temp_stability['unstable'], self.energy_mode['performance']),
            
            # Energy saving transition rules
            ctrl.Rule(self.temp_stability['stabilizing'] & self.current_temp['comfortable'], self.fan_speed['low']),
            ctrl.Rule(self.temp_stability['stabilizing'] & self.current_temp['comfortable'], self.ac_control['off']),
            
            # If air quality is poor inside, prioritize air exchange
            ctrl.Rule(self.humidity['humid'] & self.outdoor_humidity['dry'], self.exhaust_fan['high']),
            ctrl.Rule(self.humidity['humid'] & self.outdoor_humidity['dry'], self.intake_fan['high'])
        ]
    
    def calculate_temp_stability(self, current_temp):
        """Calculate temperature stability based on recent temperature readings"""
        # Add current temperature to history
        self.temp_history.append(current_temp)
        
        # Keep only the most recent readings
        if len(self.temp_history) > self.stability_window:
            self.temp_history = self.temp_history[-self.stability_window:]
        
        # If we don't have enough readings yet, consider temperature unstable
        if len(self.temp_history) < 3:
            return 0  # Unstable
            
        # Calculate stability based on:
        # 1. Variance in temperature readings
        temp_variance = np.var(self.temp_history)
        
        # 2. Proximity to target temperature
        target_diff = abs(current_temp - self.target_temp)
        
        # Combine factors to get stability score (0-100)
        variance_factor = max(0, min(100, 100 - (temp_variance * 20)))
        target_factor = max(0, min(100, 100 - (target_diff * 10)))
        
        stability_score = (variance_factor * 0.7) + (target_factor * 0.3)
        return stability_score
    
    def set_target_temperature(self, target_temp):
        """Set the target temperature for the system"""
        self.target_temp = target_temp
        print(f"Target temperature set to {target_temp}°C")
    
    def compute_control(self, current_temp, humidity, forecast_temp, outdoor_temp, outdoor_humidity):
        """Compute control outputs based on current conditions"""
        # Calculate temperature stability
        stability = self.calculate_temp_stability(current_temp)
        
        # Input the values into the control system
        self.hvac_simulation.input['current_temp'] = current_temp
        self.hvac_simulation.input['humidity'] = humidity
        self.hvac_simulation.input['forecast_temp'] = forecast_temp
        self.hvac_simulation.input['outdoor_temp'] = outdoor_temp
        self.hvac_simulation.input['outdoor_humidity'] = outdoor_humidity
        self.hvac_simulation.input['temp_stability'] = stability
        
        # Compute the result
        try:
            self.hvac_simulation.compute()
            
            # Get the outputs
            fan_speed = self.hvac_simulation.output['fan_speed']
            energy_mode = self.hvac_simulation.output['energy_mode']
            ac_control = self.hvac_simulation.output['ac_control']
            intake_fan = self.hvac_simulation.output['intake_fan']
            exhaust_fan = self.hvac_simulation.output['exhaust_fan']
            
            # Determine binary AC status (on/off)
            ac_status = "ON" if ac_control > 50 else "OFF"
            
            # Calculate energy consumption
            energy_consumption = self.calculate_energy_consumption(
                ac_control, fan_speed, intake_fan, exhaust_fan)
            
            # Calculate energy saved vs. traditional system
            energy_saved = self.calculate_energy_saved(current_temp, energy_consumption)
            
            # Store values in history
            now = datetime.datetime.now()
            self.history['timestamps'].append(now)
            self.history['current_temp'].append(current_temp)
            self.history['humidity'].append(humidity)
            self.history['outdoor_temp'].append(outdoor_temp)
            self.history['outdoor_humidity'].append(outdoor_humidity)
            self.history['forecast_temp'].append(forecast_temp)
            self.history['fan_speed'].append(fan_speed)
            self.history['energy_mode'].append(energy_mode)
            self.history['ac_control'].append(ac_control)
            self.history['intake_fan'].append(intake_fan)
            self.history['exhaust_fan'].append(exhaust_fan)
            self.history['temp_stability'].append(stability)
            self.history['energy_saved'].append(energy_saved)
            
            return {
                'fan_speed': fan_speed,
                'energy_mode': energy_mode,
                'ac_control': ac_control,
                'ac_status': ac_status,
                'intake_fan': intake_fan,
                'exhaust_fan': exhaust_fan,
                'temp_stability': stability,
                'energy_saved': energy_saved
            }
        except Exception as e:
            print(f"Error in computation: {e}")
            print("Check if input values are within defined ranges.")
            return None
    
    def calculate_energy_consumption(self, ac_control, fan_speed, intake_fan, exhaust_fan):
        """Calculate instantaneous energy consumption in kWh"""
        # Convert percentages to actual power usage
        ac_power_used = self.ac_power * (1 if ac_control > 50 else 0)
        fan_power_used = self.fan_power * (fan_speed / 100)
        intake_power_used = self.intake_power * (intake_fan / 100)
        exhaust_power_used = self.exhaust_power * (exhaust_fan / 100)
        
        # Total instantaneous power
        total_power = ac_power_used + fan_power_used + intake_power_used + exhaust_power_used
        
        return total_power
    
    def calculate_energy_saved(self, current_temp, actual_consumption):
        """Calculate energy saved compared to traditional system"""
        # Traditional system assumptions:
        # - AC runs whenever temperature is above target
        # - Fans always run at medium speed
        # - No smart air exchange
        
        traditional_ac = self.ac_power if current_temp > self.target_temp else 0
        traditional_fan = self.fan_power * 0.6  # 60% fan speed
        
        traditional_consumption = traditional_ac + traditional_fan
        energy_saved = max(0, traditional_consumption - actual_consumption)
        
        return energy_saved
    
    def view_membership_functions(self):
        """Visualize the membership functions of the fuzzy control system"""
        # Visualize key membership functions
        self.current_temp.view()
        self.outdoor_temp.view()
        self.intake_fan.view()
        self.exhaust_fan.view()
        self.ac_control.view()
        plt.show()
    
    def plot_history(self, hours=24):
        """Plot historical data for the past number of hours"""
        if len(self.history['timestamps']) == 0:
            print("No historical data available")
            return
            
        # Get data from last hours
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(hours=hours)
        
        # Filter data
        indices = [i for i, ts in enumerate(self.history['timestamps']) if ts >= cutoff]
        
        if not indices:
            print(f"No historical data available for the past {hours} hours")
            return
            
        timestamps = [self.history['timestamps'][i] for i in indices]
        temps = [self.history['current_temp'][i] for i in indices]
        outdoor_temps = [self.history['outdoor_temp'][i] for i in indices]
        forecast_temps = [self.history['forecast_temp'][i] for i in indices]
        humidity = [self.history['humidity'][i] for i in indices]
        outdoor_humidity = [self.history['outdoor_humidity'][i] for i in indices]
        fan_speed = [self.history['fan_speed'][i] for i in indices]
        intake_fan = [self.history['intake_fan'][i] for i in indices]
        exhaust_fan = [self.history['exhaust_fan'][i] for i in indices]
        ac_control = [self.history['ac_control'][i] for i in indices]
        energy_saved = [self.history['energy_saved'][i] for i in indices]
        
        # Create subplot
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
        
        # Plot temperatures
        ax1.plot(timestamps, temps, 'r-', label='Indoor Temp')
        ax1.plot(timestamps, outdoor_temps, 'b-', label='Outdoor Temp')
        ax1.plot(timestamps, forecast_temps, 'g--', label='Forecast Temp')
        ax1.set_ylabel('Temperature (°C)')
        ax1.legend(loc='upper left')
        
        # Plot humidity
        ax2.plot(timestamps, humidity, 'b-', label='Indoor Humidity')
        ax2.plot(timestamps, outdoor_humidity, 'c--', label='Outdoor Humidity')
        ax2.set_ylabel('Humidity (%)')
        ax2.legend()
        
        # Plot fan speeds
        ax3.plot(timestamps, fan_speed, 'g-', label='Internal Fan')
        ax3.plot(timestamps, intake_fan, 'y-', label='Intake Fan')
        ax3.plot(timestamps, exhaust_fan, 'm-', label='Exhaust Fan')
        ax3.set_ylabel('Fan Speed (%)')
        ax3.legend()
        
        # Plot AC control
        ax4.plot(timestamps, ac_control, 'c-', label='AC Control')
        ax4.set_ylabel('AC Control (0:OFF, 100:ON)')
        ax4.legend()
        
        # Plot energy saved
        ax5.plot(timestamps, energy_saved, 'k-', label='Energy Saved (kWh)')
        ax5.set_ylabel('Energy Saved (kWh)')
        ax5.set_xlabel('Time')
        ax5.legend()
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, days=7):
        """Generate an energy efficiency report for the past number of days"""
        if len(self.history['timestamps']) == 0:
            return "No historical data available for report generation"
            
        # Get data from last days
        now = datetime.datetime.now()
        start_date = now - datetime.timedelta(days=days)
        
        # Filter data for the period
        indices = [i for i, ts in enumerate(self.history['timestamps']) 
                  if start_date <= ts <= now]
        
        if not indices:
            return f"No historical data available for the past {days} days"
            
        # Compute averages
        avg_indoor_temp = np.mean([self.history['current_temp'][i] for i in indices])
        avg_outdoor_temp = np.mean([self.history['outdoor_temp'][i] for i in indices])
        avg_indoor_humidity = np.mean([self.history['humidity'][i] for i in indices])
        avg_outdoor_humidity = np.mean([self.history['outdoor_humidity'][i] for i in indices])
        
        # Calculate equipment usage
        ac_on_time = sum(1 for i in indices if self.history['ac_control'][i] > 50)
        fans_active_time = sum(1 for i in indices if self.history['fan_speed'][i] > 10)
        intake_active_time = sum(1 for i in indices if self.history['intake_fan'][i] > 10)
        exhaust_active_time = sum(1 for i in indices if self.history['exhaust_fan'][i] > 10)
        
        total_readings = len(indices)
        
        # Calculate percentages
        if total_readings > 0:
            ac_on_pct = (ac_on_time / total_readings) * 100
            fans_active_pct = (fans_active_time / total_readings) * 100
            intake_active_pct = (intake_active_time / total_readings) * 100
            exhaust_active_pct = (exhaust_active_time / total_readings) * 100
            
            # Natural ventilation vs mechanical cooling percentage
            natural_cooling_pct = (intake_active_time / max(1, (intake_active_time + ac_on_time))) * 100
        else:
            ac_on_pct = fans_active_pct = intake_active_pct = exhaust_active_pct = natural_cooling_pct = 0
        
        # Calculate total energy saved
        total_energy_saved = sum(self.history['energy_saved'][i] for i in indices)
        
        # Estimate cost savings (assuming $0.15 per kWh)
        cost_savings = total_energy_saved * 0.15
        
        report = f"""
        Smart HVAC System Energy Efficiency Report
        Period: {start_date.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')} ({days} days)
        
        Environmental Averages:
        - Average Indoor Temperature: {avg_indoor_temp:.1f}°C
        - Average Outdoor Temperature: {avg_outdoor_temp:.1f}°C
        - Average Indoor Humidity: {avg_indoor_humidity:.1f}%
        - Average Outdoor Humidity: {avg_outdoor_humidity:.1f}%
        
        Equipment Usage:
        - AC Active: {ac_on_pct:.1f}% of time
        - Internal Fans Active: {fans_active_pct:.1f}% of time
        - Fresh Air Intake Active: {intake_active_pct:.1f}% of time
        - Air Exhaust Active: {exhaust_active_pct:.1f}% of time
        - Natural Ventilation Used: {natural_cooling_pct:.1f}% of cooling time
        
        Energy Savings:
        - Total Energy Saved: {total_energy_saved:.2f} kWh
        - Estimated Cost Savings: ${cost_savings:.2f}
        - CO2 Reduction: {total_energy_saved * 0.7:.2f} kg (est.)
        
        Smart Strategies Used:
        - Air exchange with outdoor air when beneficial
        - Shutting down equipment when temperature stabilizes
        - Proactive adjustments based on weather forecast
        - Humidity-based control for optimal comfort
        """
        
        return report


# Example usage showing air exchange function:
def demo():
    # Initialize the system
    hvac_system = IntelligentHVACSystem()
    hvac_system.set_target_temperature(23)  # Set target temperature
    
    print("=== Smart HVAC System with Indoor-Outdoor Air Exchange ===\n")
    
    # Simulation scenarios
    scenarios = [
        {
            "name": "Hot indoor, pleasant outdoor - Natural cooling opportunity",
            "indoor_temp": 28,
            "indoor_humidity": 60,
            "outdoor_temp": 22,
            "outdoor_humidity": 40,
            "forecast_temp": 23
        },
        {
            "name": "Hot indoor, hot outdoor - Mechanical cooling needed",
            "indoor_temp": 29,
            "indoor_humidity": 65,
            "outdoor_temp": 32,
            "outdoor_humidity": 45,
            "forecast_temp": 33
        },
        {
            "name": "Comfortable indoor, cold outdoor - Retain indoor air",
            "indoor_temp": 23,
            "indoor_humidity": 50,
            "outdoor_temp": 12,
            "outdoor_humidity": 30,
            "forecast_temp": 14
        },
        {
            "name": "Humid indoor, dry outdoor - Humidity control opportunity",
            "indoor_temp": 25,
            "indoor_humidity": 75,
            "outdoor_temp": 24,
            "outdoor_humidity": 30,
            "forecast_temp": 25
        }
    ]
    
    # Run each scenario
    for scenario in scenarios:
        print(f"\n=== Scenario: {scenario['name']} ===")
        print(f"Indoor: {scenario['indoor_temp']}°C, {scenario['indoor_humidity']}% humidity")
        print(f"Outdoor: {scenario['outdoor_temp']}°C, {scenario['outdoor_humidity']}% humidity")
        print(f"Forecast: {scenario['forecast_temp']}°C\n")
        
        # Get control decisions
        result = hvac_system.compute_control(
            current_temp=scenario['indoor_temp'],
            humidity=scenario['indoor_humidity'],
            forecast_temp=scenario['forecast_temp'], 
            outdoor_temp=scenario['outdoor_temp'],
            outdoor_humidity=scenario['outdoor_humidity']
        )
        
        if result:
            # Display results
            print("Control Decisions:")
            print(f"Air Conditioner: {result['ac_status']}")
            print(f"Internal Fan Speed: {result['fan_speed']:.1f}%")
            print(f"Fresh Air Intake: {result['intake_fan']:.1f}%")
            print(f"Air Exhaust: {result['exhaust_fan']:.1f}%")

demo()