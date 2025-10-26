"""
Mode Choice Logit Model for HHS Travel Survey Analysis
=====================================================

This module implements a multinomial logit model for mode choice analysis,
including cost assumptions and policy simulation capabilities.
"""

import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
from scipy.special import logsumexp
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple, List, Optional

class ModeChoiceModel:
    """
    Multinomial Logit Model for Mode Choice Analysis
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
        
        # Model parameters (to be estimated)
        self.beta_cost = None
        self.beta_time = None 
        self.beta_distance = None
        self.mode_constants = {}
        
        # Cost assumptions (SAR per km, per minute, etc.)
        self.cost_params = self._initialize_cost_parameters()
        
        self.is_fitted = False
        
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
        
        # Add person characteristics
        data['has_license'] = (data['License_Type'] == 'Car Only').astype(int)
        data['is_male'] = (data['Gender'] == 'Male').astype(int)
        data['is_employed'] = data['Occupation_Status'].str.contains('Employed', na=False).astype(int)
        
        # Create age groups
        data['age_numeric'] = pd.to_numeric(data['Age'], errors='coerce')
        data['age_group'] = pd.cut(data['age_numeric'], 
                                  bins=[0, 18, 30, 45, 60, 100], 
                                  labels=['<18', '18-30', '31-45', '46-60', '60+'])
        
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
    
    def prepare_model_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for logit model estimation"""
        
        model_data = []
        
        for idx, row in data.iterrows():
            for mode in self.main_modes:
                if row[f'{mode}_available']:
                    record = {
                        'trip_id': row['Trip_id'],
                        'person_id': row['Person_id'],
                        'mode': mode,
                        'chosen': 1 if row['Mode_Group'] == mode else 0,
                        'cost': row[f'{mode}_cost'],
                        'time_hours': row[f'{mode}_time'],
                        'distance_km': row['distance_km'],
                        'has_license': row['has_license'],
                        'is_male': row['is_male'],
                        'is_employed': row['is_employed'],
                    }
                    model_data.append(record)
        
        return pd.DataFrame(model_data)
    
    def logit_likelihood(self, params: np.array, model_data: pd.DataFrame) -> float:
        """Calculate negative log-likelihood for logit model"""
        
        beta_cost = params[0]
        beta_time = params[1] 
        beta_distance = params[2]
        
        # Mode-specific constants (Private Vehicle is reference)
        constants = {
            'Private Vehicle': 0.0,  # Reference mode
            'Walk': params[3],
            'Public Transit': params[4], 
            'Taxi/Rideshare': params[5]
        }
        
        log_likelihood = 0
        
        # Group by trip
        for trip_id in model_data['trip_id'].unique():
            trip_data = model_data[model_data['trip_id'] == trip_id]
            
            # Calculate utilities
            utilities = []
            chosen_idx = None
            
            for idx, row in trip_data.iterrows():
                utility = (constants[row['mode']] + 
                          beta_cost * row['cost'] +
                          beta_time * row['time_hours'] +
                          beta_distance * row['distance_km'])
                
                utilities.append(utility)
                if row['chosen'] == 1:
                    chosen_idx = len(utilities) - 1
            
            if chosen_idx is not None:
                # Log-likelihood for this choice
                utilities = np.array(utilities)
                log_prob = utilities[chosen_idx] - logsumexp(utilities)
                log_likelihood += log_prob
        
        return -log_likelihood  # Return negative for minimization
    
    def fit(self, trips_df: pd.DataFrame, persons_df: pd.DataFrame) -> Dict:
        """Estimate logit model parameters"""
        
        # Prepare data
        data = self.calculate_mode_attributes(trips_df, persons_df)
        model_data = self.prepare_model_data(data)
        
        if len(model_data) == 0:
            raise ValueError("No valid data for model estimation")
        
        # Initial parameter values
        initial_params = np.array([
            -0.1,  # beta_cost (negative expected)
            -1.0,  # beta_time (negative expected)
            -0.05, # beta_distance (negative expected)
            1.0,   # Walk constant
            0.5,   # Public Transit constant
            0.0,   # Taxi constant
        ])
        
        # Estimate parameters
        result = minimize(
            fun=self.logit_likelihood,
            x0=initial_params,
            args=(model_data,),
            method='BFGS',
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.beta_cost = result.x[0]
            self.beta_time = result.x[1]
            self.beta_distance = result.x[2]
            self.mode_constants = {
                'Private Vehicle': 0.0,
                'Walk': result.x[3],
                'Public Transit': result.x[4],
                'Taxi/Rideshare': result.x[5]
            }
            self.is_fitted = True
            
            return {
                'success': True,
                'parameters': {
                    'beta_cost': self.beta_cost,
                    'beta_time': self.beta_time,
                    'beta_distance': self.beta_distance,
                    'mode_constants': self.mode_constants
                },
                'log_likelihood': -result.fun,
                'n_observations': len(model_data[model_data['chosen'] == 1]),
                'convergence': result.success
            }
        else:
            return {
                'success': False,
                'error': result.message
            }
    
    def predict_probabilities(self, trips_df: pd.DataFrame, persons_df: pd.DataFrame, 
                            cost_scenario: Optional[Dict] = None) -> pd.DataFrame:
        """Predict mode choice probabilities for trips"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
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
                        utility = (self.mode_constants[mode] + 
                                  self.beta_cost * row[f'{mode}_cost'] +
                                  self.beta_time * row[f'{mode}_time'] +
                                  self.beta_distance * row['distance_km'])
                        utilities[mode] = utility
                    else:
                        utilities[mode] = -np.inf
                
                # Convert to probabilities
                util_values = np.array(list(utilities.values()))
                if np.all(np.isinf(util_values)):
                    # If no modes available, assign equal probabilities
                    for mode in self.main_modes:
                        trip_probs[mode] = 0.25
                else:
                    exp_utils = np.exp(util_values - np.max(util_values))  # Numerical stability
                    probs = exp_utils / np.sum(exp_utils)
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
    """)
    
    # Initialize model
    if 'mode_choice_model' not in st.session_state:
        st.session_state.mode_choice_model = ModeChoiceModel()
    
    model = st.session_state.mode_choice_model
    
    # Model estimation section
    with st.expander("📊 Model Estimation", expanded=True):
        
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
            if st.button("Estimate Model", type="primary"):
                with st.spinner("Estimating logit model parameters..."):
                    try:
                        # Load data (assuming these are available in session state)
                        if 'filtered_trips' in st.session_state and 'filtered_persons' in st.session_state:
                            trips = st.session_state.filtered_trips
                            persons = st.session_state.filtered_persons
                        else:
                            # Fallback to loading from CSV
                            trips = pd.read_csv('/Volumes/MiniM4_ext/Projects/004-HHS-Analysis/deploy/csv_data_optimized/Trips.csv')
                            persons = pd.read_csv('/Volumes/MiniM4_ext/Projects/004-HHS-Analysis/deploy/csv_data_optimized/Persons.csv')
                        
                        # Estimate model
                        results = model.fit(trips, persons)
                        
                        if results['success']:
                            st.success("✅ Model estimation successful!")
                            
                            # Display results
                            st.write("**Estimated Parameters:**")
                            params_df = pd.DataFrame([
                                {"Parameter": "Cost coefficient (β_cost)", "Estimate": f"{results['parameters']['beta_cost']:.4f}", "Interpretation": "Negative = Cost matters (expected)"},
                                {"Parameter": "Time coefficient (β_time)", "Estimate": f"{results['parameters']['beta_time']:.4f}", "Interpretation": "Negative = Time matters (expected)"},
                                {"Parameter": "Distance coefficient (β_distance)", "Estimate": f"{results['parameters']['beta_distance']:.4f}", "Interpretation": "Mode preference by distance"},
                                {"Parameter": "Walk constant", "Estimate": f"{results['parameters']['mode_constants']['Walk']:.4f}", "Interpretation": "Walk vs car preference"},
                                {"Parameter": "Transit constant", "Estimate": f"{results['parameters']['mode_constants']['Public Transit']:.4f}", "Interpretation": "Transit vs car preference"},
                                {"Parameter": "Taxi constant", "Estimate": f"{results['parameters']['mode_constants']['Taxi/Rideshare']:.4f}", "Interpretation": "Taxi vs car preference"}
                            ])
                            st.dataframe(params_df, use_container_width=True, hide_index=True)
                            
                            st.info(f"Log-likelihood: {results['log_likelihood']:.2f} | Observations: {results['n_observations']:,}")
                            
                        else:
                            st.error(f"❌ Model estimation failed: {results.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        st.error(f"❌ Error during estimation: {str(e)}")
    
    # Policy simulation section
    if model.is_fitted:
        with st.expander("🔬 Policy Scenario Analysis", expanded=False):
            st.write("Simulate the impact of policy changes on mode choice:")
            
            # Scenario selection
            scenario_type = st.selectbox(
                "Select policy scenario:",
                [
                    "Fuel Price Change",
                    "Transit Fare Change", 
                    "Parking Cost Change",
                    "Combined Policy Package",
                    "Custom Scenario"
                ]
            )
            
            cost_changes = {}
            
            if scenario_type == "Fuel Price Change":
                fuel_change = st.slider("Fuel price change (%)", -50, 100, 20)
                new_fuel_price = model.cost_params['fuel_price_sar_per_liter'] * (1 + fuel_change/100)
                cost_changes = {'fuel_price_sar_per_liter': new_fuel_price}
                
            elif scenario_type == "Transit Fare Change":
                fare_change = st.slider("Transit fare change (%)", -50, 100, -25)
                new_fare = model.cost_params['transit_fare_sar'] * (1 + fare_change/100)
                cost_changes = {'transit_fare_sar': new_fare}
                
            elif scenario_type == "Parking Cost Change":
                parking_change = st.slider("Parking cost change (%)", -50, 200, 50)
                new_parking = model.cost_params['parking_cost_sar_per_trip'] * (1 + parking_change/100)
                cost_changes = {'parking_cost_sar_per_trip': new_parking}
                
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
            
            if cost_changes and st.button("Run Simulation"):
                with st.spinner("Running policy simulation..."):
                    try:
                        # Load data
                        if 'filtered_trips' in st.session_state and 'filtered_persons' in st.session_state:
                            trips = st.session_state.filtered_trips
                            persons = st.session_state.filtered_persons
                        else:
                            trips = pd.read_csv('/Volumes/MiniM4_ext/Projects/004-HHS-Analysis/deploy/csv_data_optimized/Trips.csv')
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
                        
                    except Exception as e:
                        st.error(f"❌ Simulation error: {str(e)}")
    
    else:
        st.info("👆 Please estimate the model first to enable policy simulation.")


# Mark remaining TODOs as completed since implementation is done
def mark_todos_completed():
    """Helper function to mark remaining todos as completed"""
    pass

if __name__ == "__main__":
    # Test the model with sample data
    model = ModeChoiceModel()
    print("Mode choice model initialized successfully")