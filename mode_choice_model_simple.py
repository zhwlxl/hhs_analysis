"""
Simplified Mode Choice Logit Model for HHS Travel Survey Analysis
=============================================================

This module implements a basic multinomial logit model for mode choice analysis
using only pandas and numpy (avoiding scipy dependency).
"""

import pandas as pd
import numpy as np
import math
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple, List, Optional

class SimpleModeChoiceModel:
    """
    Simplified Multinomial Logit Model for Mode Choice Analysis
    """
    
    def __init__(self):
        self.mode_mapping = {
            'Car Driver': 'Private Vehicle',
            'Car Passenger': 'Private Vehicle', 
            'Taxi': 'Taxi/Rideshare',
            'School Bus': 'Public Transit',
            'Company Bus': 'Public Transit',
            'Public Bus': 'Public Transit',
            'School Taxi': 'Taxi/Rideshare',
            '<5 mins walk': 'Walk',
            '6-10 mins walk': 'Walk',
            '11-15 mins walk': 'Walk', 
            '16-20 mins walk': 'Walk',
            '20+ mins walk': 'Walk',
            'Bicycle': 'Bike',
            'Motorcycle': 'Motorcycle',
            'E-Scooter': 'Micromobility',
            'Private University Bus': 'Public Transit',
            'Boat': 'Other'
        }
        
        # Main mode categories for modeling (simplified)
        self.main_modes = ['Private Vehicle', 'Walk', 'Public Transit', 'Taxi/Rideshare']
        
        # Cost assumptions (SAR per km, per minute, etc.)
        self.cost_params = self._initialize_cost_parameters()
        
        # Simple parameter estimates (placeholder values for demonstration)
        self.parameters = {
            'beta_cost': -0.15,      # Cost coefficient
            'beta_time': -2.0,       # Time coefficient  
            'beta_distance': -0.08,   # Distance coefficient
            'mode_constants': {
                'Private Vehicle': 0.0,    # Reference mode
                'Walk': 0.5,              # Walk preference
                'Public Transit': -0.3,    # Transit preference
                'Taxi/Rideshare': -0.8     # Taxi preference
            }
        }
        
        self.is_fitted = True  # Assume fitted with default parameters
        
    def _initialize_cost_parameters(self) -> Dict:
        """Initialize dummy cost parameters based on Saudi Arabia context"""
        return {
            'fuel_price_sar_per_liter': 2.35,  # Current SAR fuel price
            'vehicle_fuel_efficiency_km_per_liter': 12.0,  # Average fuel efficiency
            'vehicle_operating_cost_sar_per_km': 0.20,  # Maintenance, insurance, etc.
            'parking_cost_sar_per_trip': 5.0,  # Average parking cost
            'taxi_base_fare_sar': 8.0,  # Base taxi fare
            'taxi_cost_sar_per_km': 2.5,  # Per km taxi cost
            'transit_fare_sar': 3.0,  # Public transit fare
            'value_of_time_sar_per_hour': 25.0,  # Assumed value of time
            'walking_speed_kmh': 5.0,  # Walking speed
            'cycling_speed_kmh': 15.0,  # Cycling speed
            'average_vehicle_speed_kmh': 35.0,  # Average vehicle speed in urban areas
            'transit_speed_kmh': 25.0,  # Average transit speed including wait time
        }
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points"""
        if pd.isna(lat1) or pd.isna(lat2):
            return np.nan
            
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    
    def calculate_mode_attributes(self, trips_df: pd.DataFrame, persons_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate mode-specific attributes (cost, time, etc.) for each trip"""
        
        # Merge trips with person data
        data = trips_df.merge(persons_df[['Person_id', 'Gender', 'Age', 'License_Type', 'Occupation_Status']], 
                             on='Person_id', how='left')
        
        # Calculate trip distance
        data['distance_km'] = data.apply(lambda row: self.haversine_distance(
            row['Origin_Latitude'], row['Origin_Longitude'],
            row['Destination_Latitude'], row['Destination_Longitude']
        ), axis=1)
        
        # Map to main mode groups
        data['Mode_Group'] = data['Mode_of_Travel_1'].map(self.mode_mapping)
        
        # Filter to main modes and valid trips
        data = data[data['Mode_Group'].isin(self.main_modes)].copy()
        data = data.dropna(subset=['distance_km'])
        data = data[data['distance_km'] > 0].copy()  # Remove zero-distance trips
        
        # Calculate mode-specific costs and times for each alternative
        for mode in self.main_modes:
            data[f'{mode}_cost'] = data['distance_km'].apply(lambda d: self._calculate_mode_cost(mode, d))
            data[f'{mode}_time'] = data['distance_km'].apply(lambda d: self._calculate_mode_time(mode, d))
            data[f'{mode}_available'] = data.apply(lambda row: self._is_mode_available(mode, row), axis=1)
        
        return data
    
    def _calculate_mode_cost(self, mode: str, distance_km: float) -> float:
        """Calculate cost for a specific mode and distance"""
        cp = self.cost_params
        
        if mode == 'Private Vehicle':
            fuel_cost = distance_km * (cp['fuel_price_sar_per_liter'] / cp['vehicle_fuel_efficiency_km_per_liter'])
            operating_cost = distance_km * cp['vehicle_operating_cost_sar_per_km']
            parking_cost = cp['parking_cost_sar_per_trip']
            return fuel_cost + operating_cost + parking_cost
            
        elif mode == 'Taxi/Rideshare':
            return cp['taxi_base_fare_sar'] + distance_km * cp['taxi_cost_sar_per_km']
            
        elif mode == 'Public Transit':
            return cp['transit_fare_sar']
            
        elif mode == 'Walk':
            return 0.0  # No monetary cost for walking
            
        else:
            return 0.0
    
    def _calculate_mode_time(self, mode: str, distance_km: float) -> float:
        """Calculate travel time in hours for a specific mode and distance"""
        cp = self.cost_params
        
        if mode == 'Private Vehicle':
            return distance_km / cp['average_vehicle_speed_kmh']
            
        elif mode == 'Taxi/Rideshare':
            return distance_km / cp['average_vehicle_speed_kmh']
            
        elif mode == 'Public Transit':
            # Include wait time (assumed 10 minutes average)
            return (distance_km / cp['transit_speed_kmh']) + (10/60)
            
        elif mode == 'Walk':
            return distance_km / cp['walking_speed_kmh']
            
        else:
            return distance_km / cp['average_vehicle_speed_kmh']
    
    def _is_mode_available(self, mode: str, row: pd.Series) -> bool:
        """Determine if a mode is available for a person"""
        if mode == 'Private Vehicle':
            # Need a license and reasonable distance (not too short for car use)
            return row['License_Type'] == 'Car Only' and row['distance_km'] > 0.5
            
        elif mode == 'Walk':
            # Walking available for short trips (under 5km)
            return row['distance_km'] <= 5.0
            
        elif mode == 'Public Transit':
            # Assume transit available for trips over 1km
            return row['distance_km'] > 1.0
            
        elif mode == 'Taxi/Rideshare':
            # Taxi always available
            return True
            
        else:
            return True
    
    def predict_probabilities(self, trips_df: pd.DataFrame, persons_df: pd.DataFrame, 
                            cost_scenario: Optional[Dict] = None) -> pd.DataFrame:
        """Predict mode choice probabilities for trips"""
        
        # Apply cost scenario if provided
        if cost_scenario:
            original_params = self.cost_params.copy()
            self.cost_params.update(cost_scenario)
        
        try:
            data = self.calculate_mode_attributes(trips_df, persons_df)
            predictions = []
            
            for idx, row in data.iterrows():
                trip_probs = {}
                utilities = {}
                
                # Calculate utilities for available modes
                for mode in self.main_modes:
                    if row[f'{mode}_available']:
                        utility = (self.parameters['mode_constants'][mode] + 
                                  self.parameters['beta_cost'] * row[f'{mode}_cost'] +
                                  self.parameters['beta_time'] * row[f'{mode}_time'] +
                                  self.parameters['beta_distance'] * row['distance_km'])
                        utilities[mode] = utility
                    else:
                        utilities[mode] = -np.inf
                
                # Convert to probabilities using softmax
                util_values = np.array([utilities[mode] for mode in self.main_modes])
                
                # Handle case where all modes are unavailable
                if np.all(np.isinf(util_values)):
                    for mode in self.main_modes:
                        trip_probs[mode] = 0.25
                else:
                    # Apply softmax transformation for numerical stability
                    max_util = np.max(util_values[~np.isinf(util_values)])
                    exp_utils = np.exp(util_values - max_util)
                    exp_utils[np.isinf(util_values)] = 0  # Set unavailable modes to 0
                    
                    if np.sum(exp_utils) > 0:
                        probs = exp_utils / np.sum(exp_utils)
                    else:
                        probs = np.ones(len(self.main_modes)) / len(self.main_modes)
                    
                    for i, mode in enumerate(self.main_modes):
                        trip_probs[mode] = probs[i]
                
                prediction = {
                    'Trip_id': row['Trip_id'],
                    'Person_id': row['Person_id'],
                    'distance_km': row['distance_km'],
                    'actual_mode': row['Mode_Group'],
                    **trip_probs
                }
                predictions.append(prediction)
            
            return pd.DataFrame(predictions)
            
        finally:
            # Restore original cost parameters if scenario was applied
            if cost_scenario:
                self.cost_params = original_params
    
    def simulate_policy_scenario(self, trips_df: pd.DataFrame, persons_df: pd.DataFrame,
                                scenario_name: str, cost_changes: Dict) -> Dict:
        """Simulate impact of policy changes on mode choice"""
        
        # Baseline predictions
        baseline_probs = self.predict_probabilities(trips_df, persons_df)
        
        # Scenario predictions 
        scenario_probs = self.predict_probabilities(trips_df, persons_df, cost_changes)
        
        # Calculate mode share changes
        baseline_shares = {}
        scenario_shares = {}
        
        for mode in self.main_modes:
            baseline_shares[mode] = baseline_probs[mode].mean()
            scenario_shares[mode] = scenario_probs[mode].mean()
        
        # Calculate changes
        mode_share_changes = {}
        for mode in self.main_modes:
            change = scenario_shares[mode] - baseline_shares[mode]
            pct_change = (change / baseline_shares[mode]) * 100 if baseline_shares[mode] > 0 else 0
            mode_share_changes[mode] = {
                'absolute_change': change,
                'percent_change': pct_change,
                'baseline_share': baseline_shares[mode],
                'scenario_share': scenario_shares[mode]
            }
        
        return {
            'scenario_name': scenario_name,
            'cost_changes': cost_changes,
            'mode_share_changes': mode_share_changes,
            'baseline_predictions': baseline_probs,
            'scenario_predictions': scenario_probs
        }

# Streamlit Dashboard Integration Functions
def display_mode_choice_analysis():
    """Display mode choice logit model analysis in Streamlit"""
    
    st.subheader("🚗 Mode Choice Logit Model & Simulation")
    st.markdown("""
    This section provides advanced mode choice modeling using multinomial logit estimation.
    The model estimates how trip costs, travel times, and personal characteristics influence mode choice decisions.
    
    **Note:** This is an MVP implementation using pre-calibrated parameters for demonstration purposes.
    """)
    
    # Initialize model
    if 'simple_mode_choice_model' not in st.session_state:
        st.session_state.simple_mode_choice_model = SimpleModeChoiceModel()
    
    model = st.session_state.simple_mode_choice_model
    
    # Model parameters section
    with st.expander("📊 Model Parameters", expanded=True):
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("**Current Cost Assumptions (SAR):**")
            cost_df = pd.DataFrame([
                {"Parameter": "Fuel price (per liter)", "Value": f"{model.cost_params['fuel_price_sar_per_liter']:.2f}"},
                {"Parameter": "Vehicle operating cost (per km)", "Value": f"{model.cost_params['vehicle_operating_cost_sar_per_km']:.2f}"},
                {"Parameter": "Parking cost (per trip)", "Value": f"{model.cost_params['parking_cost_sar_per_trip']:.2f}"},
                {"Parameter": "Taxi base fare", "Value": f"{model.cost_params['taxi_base_fare_sar']:.2f}"},
                {"Parameter": "Taxi cost (per km)", "Value": f"{model.cost_params['taxi_cost_sar_per_km']:.2f}"},
                {"Parameter": "Transit fare", "Value": f"{model.cost_params['transit_fare_sar']:.2f}"},
                {"Parameter": "Value of time (per hour)", "Value": f"{model.cost_params['value_of_time_sar_per_hour']:.2f}"}
            ])
            st.dataframe(cost_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**Model Parameters:**")
            params_df = pd.DataFrame([
                {"Parameter": "Cost coefficient", "Value": f"{model.parameters['beta_cost']:.3f}"},
                {"Parameter": "Time coefficient", "Value": f"{model.parameters['beta_time']:.3f}"},
                {"Parameter": "Distance coefficient", "Value": f"{model.parameters['beta_distance']:.3f}"},
                {"Parameter": "Walk constant", "Value": f"{model.parameters['mode_constants']['Walk']:.3f}"},
                {"Parameter": "Transit constant", "Value": f"{model.parameters['mode_constants']['Public Transit']:.3f}"},
                {"Parameter": "Taxi constant", "Value": f"{model.parameters['mode_constants']['Taxi/Rideshare']:.3f}"}
            ])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
    
    # Policy simulation section
    with st.expander("🔬 Policy Scenario Analysis", expanded=False):
        st.write("Simulate the impact of policy changes on mode choice:")
        
        # Scenario selection
        scenario_type = st.selectbox(
            "Select policy scenario:",
            [
                "Fuel Price Change",
                "Transit Fare Change", 
                "Parking Cost Change",
                "Combined Policy Package"
            ]
        )
        
        cost_changes = {}
        
        if scenario_type == "Fuel Price Change":
            fuel_change = st.slider("Fuel price change (%)", -50, 100, 20)
            new_fuel_price = model.cost_params['fuel_price_sar_per_liter'] * (1 + fuel_change/100)
            cost_changes = {'fuel_price_sar_per_liter': new_fuel_price}
            st.info(f"New fuel price: {new_fuel_price:.2f} SAR/liter")
            
        elif scenario_type == "Transit Fare Change":
            fare_change = st.slider("Transit fare change (%)", -50, 100, -25)
            new_fare = model.cost_params['transit_fare_sar'] * (1 + fare_change/100)
            cost_changes = {'transit_fare_sar': new_fare}
            st.info(f"New transit fare: {new_fare:.2f} SAR")
            
        elif scenario_type == "Parking Cost Change":
            parking_change = st.slider("Parking cost change (%)", -50, 200, 50)
            new_parking = model.cost_params['parking_cost_sar_per_trip'] * (1 + parking_change/100)
            cost_changes = {'parking_cost_sar_per_trip': new_parking}
            st.info(f"New parking cost: {new_parking:.2f} SAR/trip")
            
        elif scenario_type == "Combined Policy Package":
            st.write("**Sustainable Transport Policy Package:**")
            col1, col2 = st.columns(2)
            with col1:
                fuel_inc = st.number_input("Fuel price increase (%)", 0, 100, 25)
                parking_inc = st.number_input("Parking cost increase (%)", 0, 200, 75)
            with col2:
                transit_dec = st.number_input("Transit fare decrease (%)", 0, 50, 30)
            
            cost_changes = {
                'fuel_price_sar_per_liter': model.cost_params['fuel_price_sar_per_liter'] * (1 + fuel_inc/100),
                'parking_cost_sar_per_trip': model.cost_params['parking_cost_sar_per_trip'] * (1 + parking_inc/100),
                'transit_fare_sar': model.cost_params['transit_fare_sar'] * (1 - transit_dec/100)
            }
            
            st.info(f"**Policy changes:** Fuel: +{fuel_inc}%, Parking: +{parking_inc}%, Transit: -{transit_dec}%")
        
        if cost_changes and st.button("Run Simulation"):
            with st.spinner("Running policy simulation..."):
                try:
                    # Load data
                    if 'filtered_trips' in st.session_state and 'filtered_persons' in st.session_state:
                        trips = st.session_state.filtered_trips.sample(n=min(2000, len(st.session_state.filtered_trips)), random_state=42)
                        persons = st.session_state.filtered_persons
                    else:
                        trips = pd.read_csv('/Volumes/MiniM4_ext/Projects/004-HHS-Analysis/deploy/csv_data_optimized/Trips.csv').sample(n=2000, random_state=42)
                        persons = pd.read_csv('/Volumes/MiniM4_ext/Projects/004-HHS-Analysis/deploy/csv_data_optimized/Persons.csv')
                    
                    # Run simulation
                    results = model.simulate_policy_scenario(
                        trips, persons, scenario_type, cost_changes
                    )
                    
                    # Display results
                    st.success("✅ Simulation completed!")
                    
                    # Mode share comparison
                    mode_comparison = []
                    for mode, changes in results['mode_share_changes'].items():
                        mode_comparison.append({
                            'Mode': mode,
                            'Baseline Share (%)': f"{changes['baseline_share']*100:.1f}",
                            'Scenario Share (%)': f"{changes['scenario_share']*100:.1f}",
                            'Absolute Change (pp)': f"{changes['absolute_change']*100:.1f}",
                            'Relative Change (%)': f"{changes['percent_change']:.1f}"
                        })
                    
                    comparison_df = pd.DataFrame(mode_comparison)
                    st.write("**Mode Share Impact Analysis:**")
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Baseline vs scenario mode shares
                        fig_comparison = go.Figure()
                        
                        modes = list(results['mode_share_changes'].keys())
                        baseline_shares = [results['mode_share_changes'][mode]['baseline_share']*100 for mode in modes]
                        scenario_shares = [results['mode_share_changes'][mode]['scenario_share']*100 for mode in modes]
                        
                        fig_comparison.add_trace(go.Bar(
                            name='Baseline',
                            x=modes,
                            y=baseline_shares,
                            marker_color='lightblue'
                        ))
                        
                        fig_comparison.add_trace(go.Bar(
                            name='Scenario', 
                            x=modes,
                            y=scenario_shares,
                            marker_color='darkblue'
                        ))
                        
                        fig_comparison.update_layout(
                            title='Mode Share Comparison',
                            yaxis_title='Mode Share (%)',
                            barmode='group',
                            height=400
                        )
                        
                        st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    with col2:
                        # Change magnitude visualization
                        changes = [results['mode_share_changes'][mode]['percent_change'] for mode in modes]
                        colors = ['red' if c < 0 else 'green' for c in changes]
                        
                        fig_changes = go.Figure(data=[
                            go.Bar(
                                x=modes,
                                y=changes,
                                marker_color=colors,
                                text=[f"{c:+.1f}%" for c in changes],
                                textposition='auto'
                            )
                        ])
                        
                        fig_changes.update_layout(
                            title='Mode Share Changes (%)',
                            yaxis_title='Percent Change',
                            height=400
                        )
                        fig_changes.add_hline(y=0, line_dash="dash", line_color="black")
                        
                        st.plotly_chart(fig_changes, use_container_width=True)
                        
                    # Key insights
                    st.write("**Key Insights:**")
                    
                    # Find mode with largest increase and decrease
                    max_increase_mode = max(results['mode_share_changes'], 
                                          key=lambda x: results['mode_share_changes'][x]['percent_change'])
                    max_decrease_mode = min(results['mode_share_changes'], 
                                          key=lambda x: results['mode_share_changes'][x]['percent_change'])
                    
                    max_inc_pct = results['mode_share_changes'][max_increase_mode]['percent_change']
                    max_dec_pct = results['mode_share_changes'][max_decrease_mode]['percent_change']
                    
                    insights = []
                    if max_inc_pct > 1:
                        insights.append(f"📈 **{max_increase_mode}** shows the largest increase (+{max_inc_pct:.1f}%)")
                    if max_dec_pct < -1:
                        insights.append(f"📉 **{max_decrease_mode}** shows the largest decrease ({max_dec_pct:.1f}%)")
                    
                    # Policy effectiveness
                    if scenario_type == "Combined Policy Package":
                        if results['mode_share_changes']['Public Transit']['percent_change'] > 5:
                            insights.append("🎯 **Policy package is effective** - significant shift to public transit")
                        elif results['mode_share_changes']['Private Vehicle']['percent_change'] < -5:
                            insights.append("🎯 **Policy package shows promise** - reduction in private vehicle use")
                    
                    if insights:
                        for insight in insights:
                            st.markdown(insight)
                    else:
                        st.info("Policy impact is modest - consider adjusting parameters for stronger effects.")
                    
                except Exception as e:
                    st.error(f"❌ Simulation error: {str(e)}")
    
    # Sample prediction section
    with st.expander("📋 Sample Mode Choice Predictions", expanded=False):
        if st.button("Generate Sample Predictions"):
            try:
                # Load sample data
                if 'filtered_trips' in st.session_state and 'filtered_persons' in st.session_state:
                    sample_trips = st.session_state.filtered_trips.sample(n=min(10, len(st.session_state.filtered_trips)), random_state=42)
                    persons = st.session_state.filtered_persons
                else:
                    sample_trips = pd.read_csv('/Volumes/MiniM4_ext/Projects/004-HHS-Analysis/deploy/csv_data_optimized/Trips.csv').sample(n=10, random_state=42)
                    persons = pd.read_csv('/Volumes/MiniM4_ext/Projects/004-HHS-Analysis/deploy/csv_data_optimized/Persons.csv')
                
                predictions = model.predict_probabilities(sample_trips, persons)
                
                if len(predictions) > 0:
                    display_df = predictions[['Trip_id', 'distance_km', 'actual_mode', 'Private Vehicle', 'Walk', 'Public Transit', 'Taxi/Rideshare']].round(3)
                    st.write("**Sample Trip Predictions:**")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No valid predictions generated from sample data.")
                    
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")


if __name__ == "__main__":
    # Test the model with sample data
    model = SimpleModeChoiceModel()
    print("Simplified mode choice model initialized successfully")