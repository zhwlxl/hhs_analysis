import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.stats import gamma, lognorm, expon
import os
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('absl').setLevel(logging.ERROR)

# Set page configuration
st.set_page_config(
    page_title="Demo HHS Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimal top spacing
st.markdown("""
<style>
    /* Minimize top padding in sidebar */
    section[data-testid="stSidebar"] > div {
        padding-top: 0rem;
    }
    
    /* Reduce main container top padding */
    .block-container {
        padding-top: 0.5rem;
    }
    
    /* Make sidebar headers very compact */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        margin-top: 0.2rem;
        margin-bottom: 0.3rem;
    }
    
    /* Reduce sidebar content margins */
    section[data-testid="stSidebar"] .element-container {
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    /* First element in sidebar should have no top margin */
    section[data-testid="stSidebar"] > div > div:first-child {
        margin-top: 0;
        padding-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Define mappings for categorization
OCCUPATION_TO_STATUS = {
    'Full Time Employed': 'W',
    'Part Time Employed': 'W',
    'Self Employed': 'W',
    'Student - Adult': 'S',
    'Student - University': 'S',
    'Student - College': 'S',
    'Student - University: Full-Time (does not work)': 'S',
    'Student - University: Full-Time - Works Part-Time': 'S',
    'Student - University: Part-Time - Works Part-Time': 'S',
    'Student - College: Full-Time (does not work)': 'S',
    'Student - College: Full-Time - Works Part-Time': 'S',
    'Student - College: Part-Time - Works Part-Time': 'S',
    'Student - Secondary School': 'SS',
    'Student - Primary School': 'SP',
    'Student - Elementary School': 'SP',
    'House Person': 'NW',
    'Retired': 'NW',
    'Not Employed': 'NW',
    'Under 6 Years': 'NW',
    'Pre-school': 'NW',
    'Unemployed': 'NW'
}

# Activity mappings
ACTIVITY_MAPPING = {
    'Be at Home': 'Home',
    'Be at Work': 'Work',
    'Education': 'School',
    'Business': 'Business',
    'Shopping': 'Shopping',
    'Drop off': 'Drop-off',
    'Drop off Education': 'Drop-off Edu',
    'Other': 'Others',
    'Recreation': 'Others',
    'Personal Business': 'Business',
    'Medical': 'Others',
    'Religious': 'Others',
    'Social': 'Others'
}

# Function to categorize occupation status
def categorize_status(occupation):
    # Check if occupation contains student-related keywords
    if pd.notna(occupation):
        occupation_lower = str(occupation).lower()
        if 'student' in occupation_lower:
            if any(term in occupation_lower for term in ['adult', 'university', 'college']):
                return 'S'
            elif 'secondary' in occupation_lower:
                return 'SS'
            elif 'primary' in occupation_lower:
                return 'SP'
    return OCCUPATION_TO_STATUS.get(occupation, 'NW')

# Function to categorize nationality
def categorize_nationality(nationality_type):
    if nationality_type == 'Saudi':
        return 'Saudi Arab'
    elif nationality_type == 'Non-Saudi Arab':
        return 'Non-Saudi Arab'
    elif nationality_type == 'Non-Saudi Non-Arab':
        return 'Non-Saudi Non-Arab'
    else:
        return 'Unknown'

# Function to standardize trip purposes
def standardize_purpose(purpose):
    if pd.isna(purpose):
        return 'Others'
    
    # Check for exact matches first
    if purpose in ACTIVITY_MAPPING:
        return ACTIVITY_MAPPING[purpose]
    
    # Check for partial matches
    purpose_lower = str(purpose).lower()
    if 'home' in purpose_lower:
        return 'Home'
    elif 'work' in purpose_lower:
        return 'Work'
    elif 'education' in purpose_lower or 'school' in purpose_lower:
        return 'School'
    elif 'business' in purpose_lower:
        return 'Business'
    elif 'shopping' in purpose_lower:
        return 'Shopping'
    elif 'drop' in purpose_lower:
        if 'education' in purpose_lower:
            return 'Drop-off Edu'
        else:
            return 'Drop-off'
    else:
        return 'Others'

# Function to determine car availability based on vehicle ownership and license
def determine_car_availability(persons_df, vehicles_df=None):
    """Determine if a person has access to a car based on license and household vehicles"""
    persons_df['Car_Available'] = 'NCA'  # Default to no car
    
    # Check if person has a car license
    has_car_license = persons_df['License_Type'].str.contains('Car', case=False, na=False)
    
    # For now, assume those with car license have car access
    persons_df.loc[has_car_license, 'Car_Available'] = 'CA'
    
    # Additional logic can be added here if vehicle data is available
    
    return persons_df

# Function to calculate tours from trips
def calculate_tours(trips_df):
    """Calculate tours (round trips from home) from trip data"""
    tours = []
    
    # Group trips by person
    for person_id in trips_df['Person_id'].unique():
        person_trips = trips_df[trips_df['Person_id'] == person_id].sort_values('Trip_Number')
        
        if len(person_trips) == 0:
            tours.append({'Person_id': person_id, 'tour_count': 0})
            continue
        
        tour_count = 0
        home_to_out = 0
        out_to_home = 0
        
        for _, trip in person_trips.iterrows():
            origin_purpose = standardize_purpose(trip['Trip_Origin_Purpose'])
            dest_purpose = standardize_purpose(trip['Trip_Destination_Purpose'])
            
            if origin_purpose == 'Home' and dest_purpose != 'Home':
                home_to_out += 1
            elif origin_purpose != 'Home' and dest_purpose == 'Home':
                out_to_home += 1
        
        # A tour is completed when someone goes out from home and returns
        tour_count = min(home_to_out, out_to_home)
        
        tours.append({'Person_id': person_id, 'tour_count': tour_count})
    
    return pd.DataFrame(tours)

# Function to assign income levels based on some criteria
def assign_income_levels(persons_df, households_df):
    """Assign income levels based on available data"""
    # This is a placeholder - you should replace with actual income data if available
    # For now, using occupation type as a proxy
    
    income_mapping = {
        'W': ['3H', '2M', '1L'],  # Workers have varied income
        'NW': ['2M', '1L', '1L'],  # Non-workers tend to lower income
        'S': ['2M', '1L', '1L'],   # Students
        'SS': ['2M', '1L', '1L'],  # Secondary students
        'SP': ['2M', '1L', '1L']   # Primary students
    }
    
    persons_df['Income_Level'] = persons_df['Status'].apply(
        lambda x: np.random.choice(income_mapping.get(x, ['2M', '1L', '1L']), 
                                 p=[0.3, 0.5, 0.2] if x == 'W' else [0.2, 0.5, 0.3])
    )
    
    return persons_df

# Load and process data
@st.cache_data
def load_and_process_data(sample_size=None):
    """Load and process the household travel survey data"""
    
    try:
        # Load data
        households_df = pd.read_csv('csv_data/Jaddah_HHS.csv')
        persons_df = pd.read_csv('csv_data/Jaddah_Persons.csv')
        trips_df = pd.read_csv('csv_data/Jaddah_Trips.csv')
        
        # Try to load vehicles data if available
        try:
            vehicles_df = pd.read_csv('csv_data/Jaddah_Vehicles.csv')
        except:
            vehicles_df = None
            st.info("Vehicle data not found, using simplified car availability determination")
        
        # Apply sample size filter if specified
        if sample_size and sample_size != 'All':
            sample_households = households_df['Family_Number'].unique()[:int(sample_size)]
            households_df = households_df[households_df['Family_Number'].isin(sample_households)]
            persons_df = persons_df[persons_df['Family_Number'].isin(sample_households)]
            trips_df = trips_df[trips_df['Family_Number'].isin(sample_households)]
        
        # Process persons data
        persons_df['Status'] = persons_df['Occupation_Status'].apply(categorize_status)
        persons_df['Nationality_Group'] = persons_df['Nationality_Type'].apply(categorize_nationality)
        
        # Determine car availability
        persons_df = determine_car_availability(persons_df, vehicles_df)
        
        # Assign income levels
        persons_df = assign_income_levels(persons_df, households_df)
        
        # Standardize trip purposes
        trips_df['Trip_Origin_Purpose_Std'] = trips_df['Trip_Origin_Purpose'].apply(standardize_purpose)
        trips_df['Trip_Destination_Purpose_Std'] = trips_df['Trip_Destination_Purpose'].apply(standardize_purpose)
        
        # Ensure Trip_id exists
        if 'Trip_id' not in trips_df.columns:
            trips_df['Trip_id'] = trips_df.index
        
        # Calculate trip counts per person
        trip_counts = trips_df.groupby('Person_id').size().reset_index(name='trip_count')
        
        # Calculate tour counts
        tours_df = calculate_tours(trips_df)
        
        # Merge data
        persons_df = persons_df.merge(trip_counts, on='Person_id', how='left')
        persons_df = persons_df.merge(tours_df, on='Person_id', how='left')
        
        # Fill NaN values
        persons_df['trip_count'] = persons_df['trip_count'].fillna(0)
        persons_df['tour_count'] = persons_df['tour_count'].fillna(0)
        
        return households_df, persons_df, trips_df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Calculate mobility rates with proper grouping
def calculate_mobility_rates(persons_df, groupby_cols):
    """Calculate trip and tour rates by specified grouping columns"""
    
    # Filter out persons who didn't want to travel or had no trips recorded
    active_persons = persons_df[persons_df['Trips_Done'] != "Don't want to go out"].copy()
    
    # Group and calculate rates
    grouped = active_persons.groupby(groupby_cols).agg({
        'Person_id': 'count',
        'trip_count': 'sum',
        'tour_count': 'sum'
    }).reset_index()
    
    grouped.columns = list(groupby_cols) + ['person_count', 'total_trips', 'total_tours']
    
    # Calculate rates
    grouped['trip_rate'] = grouped['total_trips'] / grouped['person_count']
    grouped['tour_rate'] = grouped['total_tours'] / grouped['person_count']
    
    return grouped

# Enhanced trip distribution calculation
def calculate_trip_distribution(trips_df, persons_df, by_columns=['Status']):
    """Calculate trip distribution by purpose with flexible grouping"""
    
    # Merge trip data with person data
    trips_with_info = trips_df.merge(
        persons_df[['Person_id'] + by_columns + ['Gender', 'Nationality_Group', 'Car_Available', 'Income_Level']], 
        on='Person_id'
    )
    
    # Group by specified columns and destination purpose
    group_cols = by_columns + ['Trip_Destination_Purpose_Std']
    distribution = trips_with_info.groupby(group_cols).size().reset_index(name='trip_count')
    
    # Calculate percentages within each group
    for col in by_columns:
        total_by_group = distribution.groupby(col)['trip_count'].transform('sum')
        distribution[f'percentage_by_{col}'] = (distribution['trip_count'] / total_by_group * 100).round(2)
    
    return distribution

# Enhanced mode share calculation
def calculate_mode_share(trips_df, persons_df, by_columns=['Status']):
    """Calculate mode share by market segment with flexible grouping"""
    
    # Merge trip data with person data
    trips_with_info = trips_df.merge(
        persons_df[['Person_id'] + by_columns + ['Gender', 'Nationality_Group', 'Car_Available', 'Income_Level']], 
        on='Person_id'
    )
    
    # Clean mode data
    trips_with_info['Mode_Clean'] = trips_with_info['Mode_of_Travel_1'].fillna('Unknown')
    
    # Group by specified columns and mode
    group_cols = by_columns + ['Mode_Clean']
    mode_share = trips_with_info.groupby(group_cols).size().reset_index(name='trip_count')
    
    # Calculate percentages
    for col in by_columns:
        total_by_group = mode_share.groupby(col)['trip_count'].transform('sum')
        mode_share[f'percentage_by_{col}'] = (mode_share['trip_count'] / total_by_group * 100).round(2)
    
    return mode_share

# Create summary table matching the format shown in the image
def create_mobility_summary_table(persons_df):
    """Create a summary table similar to the one shown in the image"""
    
    # Group by Status, Gender, and Car Availability
    summary = persons_df.groupby(['Status', 'Gender', 'Car_Available']).agg({
        'tour_count': 'sum',
        'trip_count': 'sum',
        'Person_id': 'count'
    }).reset_index()
    
    # Rename columns
    summary.columns = ['Status', 'Gender', 'Car_Available', 'Sum_of_Tour', 'Sum_of_Trips', 'Person_Count']
    
    # Calculate rates
    summary['Tour_rate'] = (summary['Sum_of_Tour'] / summary['Person_Count']).round(4)
    summary['Trip_rate'] = (summary['Sum_of_Trips'] / summary['Person_Count']).round(4)
    
    # Select and reorder columns
    summary = summary[['Status', 'Gender', 'Car_Available', 'Sum_of_Tour', 'Tour_rate', 'Sum_of_Trips', 'Trip_rate']]
    
    # Add grand total
    grand_total = pd.DataFrame({
        'Status': ['Grand Total'],
        'Gender': [''],
        'Car_Available': [''],
        'Sum_of_Tour': [summary['Sum_of_Tour'].sum()],
        'Tour_rate': [persons_df['tour_count'].sum() / len(persons_df)],
        'Sum_of_Trips': [summary['Sum_of_Trips'].sum()],
        'Trip_rate': [persons_df['trip_count'].sum() / len(persons_df)]
    })
    
    summary = pd.concat([summary, grand_total], ignore_index=True)
    
    return summary

# Removed Gemini initialization and assistant functions

# Main dashboard
def main():
    st.title("🚗 Demo Household Travel Survey Analysis Dashboard")
    st.markdown("---")
    
    # Removed chatbot initialization
    
    # Sidebar configuration
    with st.sidebar:
        # Dashboard Controls header flush with top
        st.markdown("<h2 style='margin-top: -10px; margin-bottom: 0.3rem; padding-top: 0;'>📊 Dashboard Controls</h2>", unsafe_allow_html=True)
        
        st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
        
        # Sample size filter with compact spacing
        st.markdown("<h3 style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>Data Sample</h3>", unsafe_allow_html=True)
        sample_size = st.selectbox(
            "Select Sample Size",
            ['1000', '2000', '3000', 'All'],
            index=3,
            help="Filter the analysis by number of households"
        )
        # Store in session state for chatbot context
        st.session_state['sample_size'] = sample_size
        
        # Load data
        with st.spinner('Loading data...'):
            households_df, persons_df, trips_df = load_and_process_data(sample_size)
        
        if households_df is None:
            st.error("Failed to load data. Please check the data files.")
            return
        
        st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>🔍 Filters</h3>", unsafe_allow_html=True)
        
        # Market group filters
        status_filter = st.multiselect(
            "Market Group (Status)",
            options=['W', 'NW', 'S', 'SS', 'SP'],
            default=['W', 'NW', 'S', 'SS', 'SP'],
            help="W=Worker, NW=Non-Worker, S=Student, SS=Secondary School, SP=Primary School"
        )
        
        gender_filter = st.multiselect(
            "Gender",
            options=['Male', 'Female'],
            default=['Male', 'Female']
        )
        
        nationality_filter = st.multiselect(
            "Nationality",
            options=['Saudi Arab', 'Non-Saudi Arab', 'Non-Saudi Non-Arab'],
            default=['Saudi Arab', 'Non-Saudi Arab', 'Non-Saudi Non-Arab']
        )
        
        car_filter = st.multiselect(
            "Car Availability",
            options=['CA', 'NCA'],
            default=['CA', 'NCA'],
            help="CA=Car Available, NCA=No Car Available"
        )
        
        income_filter = st.multiselect(
            "Income Level",
            options=['1L', '2M', '3H'],
            default=['1L', '2M', '3H'],
            help="1L=Low Income, 2M=Medium Income, 3H=High Income"
        )
    
    # Apply filters
    filtered_persons = persons_df[
        (persons_df['Status'].isin(status_filter)) &
        (persons_df['Gender'].isin(gender_filter)) &
        (persons_df['Nationality_Group'].isin(nationality_filter)) &
        (persons_df['Car_Available'].isin(car_filter)) &
        (persons_df['Income_Level'].isin(income_filter))
    ]
    
    filtered_trip_ids = filtered_persons['Person_id'].unique()
    filtered_trips = trips_df[trips_df['Person_id'].isin(filtered_trip_ids)]
    
    # Main content area
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Households", f"{households_df['Family_Number'].nunique():,}")
    with col2:
        st.metric("Total Persons", f"{len(filtered_persons):,}")
    with col3:
        st.metric("Total Trips", f"{len(filtered_trips):,}")
    with col4:
        avg_trip_rate = filtered_persons['trip_count'].sum() / len(filtered_persons) if len(filtered_persons) > 0 else 0
        st.metric("Avg Trip Rate", f"{avg_trip_rate:.2f}")
    with col5:
        avg_tour_rate = filtered_persons['tour_count'].sum() / len(filtered_persons) if len(filtered_persons) > 0 else 0
        st.metric("Avg Tour Rate", f"{avg_tour_rate:.2f}")
    
    st.markdown("---")
    
    # Add map showing household locations
    with st.container():
        st.subheader("🗺️ Geographic Distribution of Selected Households by Nationality")
        
        # Get household locations for filtered persons
        household_ids = filtered_persons['Family_Number'].unique()
        
        # Merge household data with person data to get nationality
        household_persons = filtered_persons.groupby('Family_Number').agg({
            'Nationality_Group': 'first'  # Take the first person's nationality as household nationality
        }).reset_index()
        
        filtered_households = households_df[households_df['Family_Number'].isin(household_ids)]
        filtered_households = filtered_households.merge(household_persons, on='Family_Number', how='left')
        
        # Create map if location data exists
        if 'House_Address_Latitude' in filtered_households.columns and 'House_Address_Longitude' in filtered_households.columns:
            # Clean location data
            map_data = filtered_households[['Family_Number', 'House_Address_Latitude', 'House_Address_Longitude', 'Nationality_Group']].copy()
            map_data = map_data.dropna(subset=['House_Address_Latitude', 'House_Address_Longitude'])
            
            if len(map_data) > 0:
                # Define colors for nationalities
                nationality_colors = {
                    'Saudi Arab': '#FF4B4B',
                    'Non-Saudi Arab': '#4B9BFF',
                    'Non-Saudi Non-Arab': '#4BFF9B',
                    'Unknown': '#808080'
                }
                
                # Create the map
                fig_map = px.scatter_mapbox(
                    map_data,
                    lat='House_Address_Latitude',
                    lon='House_Address_Longitude',
                    color='Nationality_Group',
                    hover_data=['Family_Number'],
                    color_discrete_map=nationality_colors,
                    zoom=10,
                    height=400,
                    title=f"Home Locations of {len(map_data)} Households by Nationality"
                )
                
                fig_map.update_layout(
                    mapbox_style="carto-positron",
                    margin={"r":0,"t":30,"l":0,"b":0},
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255, 255, 255, 0.8)"
                    )
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("No valid location data available for the selected households.")
        else:
            st.info("Location data columns not found in the household dataset.")
    
    st.markdown("---")
    
    # Create custom CSS for bigger, more prominent tabs with full width
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        width: 100%;
    }
    .stTabs [data-baseweb="tab"] {
        height: 90px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        font-size: 20px;
        font-weight: bold;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e0e0e0;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
        border-color: #FF4B4B !important;
        box-shadow: 0 4px 6px rgba(255, 75, 75, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create three main analysis tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 LEVEL 1\nMobility Rates\n(How often they travel)",
        "🗺️ LEVEL 2\nTrip Distribution\n(Where they travel)",
        "🚗 LEVEL 3\nMode Choice\n(How they travel)"
    ])
    
    # Track active tab for chatbot context
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 'Level 1: Mobility Rates'
    
    with tab1:
        st.session_state.active_tab = 'Level 1: Mobility Rates'
        st.header("Mobility Rate Analysis")
        st.markdown("This analysis shows how frequently different market segments travel (trip and tour rates).")
        
        # Create summary table
        st.subheader("Summary Table by Market Group")
        summary_table = create_mobility_summary_table(filtered_persons)
        
        # Style the dataframe
        styled_summary = summary_table.style.format({
            'Sum_of_Tour': '{:,.0f}',
            'Tour_rate': '{:.4f}',
            'Sum_of_Trips': '{:,.0f}',
            'Trip_rate': '{:.4f}'
        })
        
        # Highlight the grand total row
        def highlight_total(row):
            if row['Status'] == 'Grand Total':
                return ['background-color: #e8e8e8'] * len(row)
            return [''] * len(row)
        
        styled_summary = styled_summary.apply(highlight_total, axis=1)
        st.dataframe(styled_summary, use_container_width=True, height=400)
        
        # Default grouping options for visualizations
        groupby_options = ['Status', 'Gender', 'Car_Available']
        
        if groupby_options:
            mobility_rates = calculate_mobility_rates(filtered_persons, groupby_options)
            
            # Visualizations
            st.subheader("Mobility Rate Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Trip rate chart
                primary_group = groupby_options[0] if groupby_options else 'Status'
                color_group = groupby_options[1] if len(groupby_options) > 1 else None
                
                fig_trip = px.bar(
                    mobility_rates, 
                    x=primary_group, 
                    y='trip_rate',
                    color=color_group,
                    title="Trip Rates by Market Segment",
                    labels={'trip_rate': 'Trip Rate (trips/person)', primary_group: primary_group},
                    text='trip_rate'
                )
                fig_trip.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                st.plotly_chart(fig_trip, use_container_width=True)
            
            with col2:
                # Tour rate chart
                fig_tour = px.bar(
                    mobility_rates, 
                    x=primary_group, 
                    y='tour_rate',
                    color=color_group,
                    title="Tour Rates by Market Segment",
                    labels={'tour_rate': 'Tour Rate (tours/person)', primary_group: primary_group},
                    text='tour_rate'
                )
                fig_tour.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                st.plotly_chart(fig_tour, use_container_width=True)
            
            # Remove the detailed mobility rates table section
    
    # Additional Insights for Mobility Tab
    with tab1:
        st.markdown("---")
        with st.expander("📈 Additional Mobility Insights", expanded=False):
            try:
                col1, col2 = st.columns(2)
                
                with col1:
                    cross_var1 = st.selectbox("Select first variable:", 
                                             ['Status', 'Gender', 'Nationality_Group', 'Car_Available', 'Income_Level'],
                                             key='cross1_mob')
                with col2:
                    cross_var2 = st.selectbox("Select second variable:", 
                                             ['Gender', 'Nationality_Group', 'Car_Available', 'Income_Level', 'Status'],
                                             key='cross2_mob')
                
                if cross_var1 != cross_var2:
                    # Create cross-tabulation
                    cross_mobility = calculate_mobility_rates(filtered_persons, [cross_var1, cross_var2])
                    
                    if len(cross_mobility) > 0:
                        # Pivot for heatmap
                        pivot_cross = cross_mobility.pivot_table(
                            values='trip_rate',
                            index=cross_var1,
                            columns=cross_var2,
                            aggfunc='mean'
                        )
                        
                        # Create heatmap
                        fig_heat = px.imshow(
                            pivot_cross,
                            labels=dict(x=cross_var2, y=cross_var1, color="Trip Rate"),
                            title=f"Trip Rate Heatmap: {cross_var1} vs {cross_var2}",
                            aspect="auto",
                            color_continuous_scale='RdYlBu_r'
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)
                    else:
                        st.info("No data available for the selected combination.")
                else:
                    st.warning("Please select different variables for comparison.")
            except Exception as e:
                st.info("Unable to generate cross-tabulation analysis. Please try different variable combinations.")
    
    with tab2:
        st.session_state.active_tab = 'Level 2: Trip Distribution'
        st.header("Trip Distribution Analysis")
        st.markdown("This analysis shows where people travel and for what activities.")
        
        # Default grouping for trip distribution
        dist_groupby = 'Status'
        
        # Calculate trip distribution
        trip_distribution = calculate_trip_distribution(filtered_trips, filtered_persons, [dist_groupby])
        # Sunburst chart - Make it bigger
        fig_dist = px.sunburst(
            trip_distribution,
            path=[dist_groupby, 'Trip_Destination_Purpose_Std'],
            values='trip_count',
            title=f"Trip Distribution by {dist_groupby} and Activity Purpose",
            color='trip_count',
            hover_data=[f'percentage_by_{dist_groupby}'],
            color_continuous_scale='Blues',
            height=600  # Increased height
        )
        fig_dist.update_traces(textinfo='label+percent parent')
        fig_dist.update_layout(
            font=dict(size=14),
            title_font_size=20
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Activity distribution details
        st.subheader("Activity Distribution by Market Segment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall purpose distribution - Make it bigger
            purposes = trip_distribution.groupby('Trip_Destination_Purpose_Std')['trip_count'].sum().sort_values(ascending=False)
            
            fig_purpose = px.pie(
                values=purposes.values,
                names=purposes.index,
                title="Overall Activity Distribution",
                hole=0.4,
                height=500  # Increased height
            )
            fig_purpose.update_traces(textposition='inside', textinfo='percent+label')
            fig_purpose.update_layout(
                font=dict(size=14),
                title_font_size=18
            )
            st.plotly_chart(fig_purpose, use_container_width=True)
        
        with col2:
            # Stacked bar chart by segment
            pivot_dist = trip_distribution.pivot_table(
                values=f'percentage_by_{dist_groupby}', 
                index='Trip_Destination_Purpose_Std', 
                columns=dist_groupby, 
                fill_value=0
            )
            
            fig_stacked = px.bar(
                pivot_dist.T,
                title=f"Activity Distribution by {dist_groupby} (%)",
                labels={'value': 'Percentage (%)', 'index': dist_groupby},
                height=400
            )
            fig_stacked.update_layout(barmode='stack', showlegend=True)
            st.plotly_chart(fig_stacked, use_container_width=True)
        
        # Trip Length Distribution Analysis
        st.subheader("Trip Length Distribution Analysis")
        
        # Calculate trip distances if coordinates are available
        if all(col in trips_df.columns for col in ['Origin_Latitude', 'Origin_Longitude', 'Destination_Latitude', 'Destination_Longitude']):
            # Calculate Haversine distance for each trip
            def haversine_distance(lat1, lon1, lat2, lon2):
                """Calculate distance between two points on Earth in kilometers"""
                R = 6371  # Earth's radius in kilometers
                
                lat1_rad = np.radians(lat1)
                lat2_rad = np.radians(lat2)
                delta_lat = np.radians(lat2 - lat1)
                delta_lon = np.radians(lon2 - lon1)
                
                a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                
                return R * c
            
            # Calculate distances for filtered trips
            trips_with_distance = filtered_trips.copy()
            trips_with_distance = trips_with_distance.dropna(subset=['Origin_Latitude', 'Origin_Longitude', 'Destination_Latitude', 'Destination_Longitude'])
            
            if len(trips_with_distance) > 0:
                trips_with_distance['Distance_km'] = trips_with_distance.apply(
                    lambda row: haversine_distance(
                        row['Origin_Latitude'], row['Origin_Longitude'],
                        row['Destination_Latitude'], row['Destination_Longitude']
                    ), axis=1
                )
                
                # Ensure we have Trip_id column
                if 'Trip_id' not in trips_with_distance.columns:
                    trips_with_distance['Trip_id'] = trips_with_distance.index
                
                # Merge with person data to get status
                trips_with_distance = trips_with_distance.merge(
                    filtered_persons[['Person_id', 'Status', 'Gender', 'Car_Available']], 
                    on='Person_id'
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create trip length distribution chart
                    # Define distance bins
                    bins = [0, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                    labels = ['0-2', '2-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50']
                    
                    trips_with_distance['Distance_Bin'] = pd.cut(trips_with_distance['Distance_km'], bins=bins, labels=labels, include_lowest=True)
                    
                    # Group by distance bin and status
                    distance_dist = trips_with_distance.groupby(['Distance_Bin', 'Status']).size().reset_index(name='Count')
                    
                    # Create line chart similar to the image
                    fig_distance = px.line(
                        distance_dist,
                        x='Distance_Bin',
                        y='Count',
                        color='Status',
                        title='Trip Length Distribution by Market Segment',
                        labels={'Distance_Bin': 'Distance (km)', 'Count': 'Number of Trips'},
                        markers=True
                    )
                    
                    fig_distance.update_layout(
                        xaxis_tickangle=-45,
                        height=400
                    )
                    
                    st.plotly_chart(fig_distance, use_container_width=True)
                
                with col2:
                    # Average trip length by market segment
                    avg_distance = trips_with_distance.groupby('Status')['Distance_km'].agg(['mean', 'median', 'count']).round(2)
                    avg_distance.columns = ['Average Distance (km)', 'Median Distance (km)', 'Number of Trips']
                    avg_distance = avg_distance.reset_index()
                    
                    st.subheader("Average Trip Length by Market Segment")
                    st.dataframe(
                        avg_distance.style.format({
                            'Average Distance (km)': '{:.2f}',
                            'Median Distance (km)': '{:.2f}',
                            'Number of Trips': '{:,.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Overall statistics
                    st.metric("Overall Average Trip Distance", f"{trips_with_distance['Distance_km'].mean():.2f} km")
                    st.metric("Overall Median Trip Distance", f"{trips_with_distance['Distance_km'].median():.2f} km")
                
                # Add trip length distribution by activity purpose
                st.subheader("Trip Length Distribution by Activity Purpose")
                
                # Add standardized destination purpose if we have the original purpose column
                has_purpose_data = False
                if 'Trip_Destination_Purpose' in trips_with_distance.columns:
                    trips_with_distance['Trip_Destination_Purpose_Std'] = trips_with_distance['Trip_Destination_Purpose'].apply(standardize_purpose)
                    has_purpose_data = True
                elif 'Trip_Destination_Purpose_Std' in trips_with_distance.columns:
                    has_purpose_data = True
                
                if has_purpose_data:
                    # Group by distance bin and purpose
                    distance_by_purpose = trips_with_distance.groupby(['Distance_Bin', 'Trip_Destination_Purpose_Std']).size().reset_index(name='Count')
                    
                    # Create line chart for trip length by purpose
                    fig_purpose_distance = px.line(
                        distance_by_purpose,
                        x='Distance_Bin',
                        y='Count',
                        color='Trip_Destination_Purpose_Std',
                        title='Trip Length Distribution by Activity Purpose',
                        labels={'Distance_Bin': 'Distance (km)', 'Count': 'Number of Trips', 'Trip_Destination_Purpose_Std': 'Activity'},
                        markers=True,
                        height=500
                    )
                    
                    fig_purpose_distance.update_layout(
                        xaxis_tickangle=-45,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=1.01
                        )
                    )
                    
                    st.plotly_chart(fig_purpose_distance, use_container_width=True)
                    
                    # Average trip length by purpose - Create transposed table
                    st.subheader("Average Distance by Activity Purpose")
                    
                    # Calculate statistics by purpose
                    avg_by_purpose = trips_with_distance.groupby('Trip_Destination_Purpose_Std')['Distance_km'].agg(['mean', 'median']).round(2)
                    
                    # Transpose the dataframe
                    avg_transposed = avg_by_purpose.T
                    avg_transposed.index = ['Average (km)', 'Median (km)']
                    
                    # Display the transposed table
                    st.dataframe(
                        avg_transposed.style.format("{:.2f}"),
                        use_container_width=True
                    )
                else:
                    st.info("Trip destination purpose data not available for distance analysis.")
                
                # Trip Length Distribution Function Estimation
                st.subheader("Trip Length Distribution Function Estimation by Purpose")
                
                if has_purpose_data:
                    # Define distribution functions
                    def negative_exponential(x, a, b):
                        """Negative exponential: F(x) = a * exp(-b * x)"""
                        return a * np.exp(-b * x)
                    
                    def log_normal(x, a, b, c):
                        """Log-normal: F(x) = a * exp(-(log(x) - b)^2 / (2 * c^2))"""
                        with np.errstate(divide='ignore', invalid='ignore'):
                            result = np.where(x > 0, a * np.exp(-(np.log(x) - b)**2 / (2 * c**2)), 0)
                        return result
                    
                    def gamma_dist(x, a, b, c):
                        """Gamma distribution: F(x) = a * (x^b) * exp(-c * x)"""
                        return a * (x**b) * np.exp(-c * x)
                    
                    # Get unique purposes
                    purposes = trips_with_distance['Trip_Destination_Purpose_Std'].unique()
                    
                    # Create tabs for different purposes
                    purpose_tabs = st.tabs(purposes.tolist())
                    
                    for idx, purpose in enumerate(purposes):
                        with purpose_tabs[idx]:
                            # Filter data for this purpose
                            purpose_data = trips_with_distance[trips_with_distance['Trip_Destination_Purpose_Std'] == purpose]
                            distances = purpose_data['Distance_km'].values
                            
                            # Filter distances up to 100km
                            distances = distances[distances <= 100]
                            
                            if len(distances) == 0:
                                st.info(f"No trips found for {purpose} within 100km range.")
                                continue
                            
                            # Create histogram data
                            hist, bin_edges = np.histogram(distances, bins=30, density=True)
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                            
                            # Fit different distribution functions
                            results = {}
                            
                            # Negative Exponential
                            try:
                                popt_exp, _ = curve_fit(negative_exponential, bin_centers, hist, p0=[1, 0.1], maxfev=5000)
                                y_exp = negative_exponential(bin_centers, *popt_exp)
                                r2_exp = 1 - np.sum((hist - y_exp)**2) / np.sum((hist - hist.mean())**2)
                                results['Negative Exponential'] = {
                                    'params': popt_exp,
                                    'r2': r2_exp,
                                    'formula': f'F(x) = {popt_exp[0]:.3f} * exp(-{popt_exp[1]:.3f} * x)',
                                    'y_fit': y_exp
                                }
                            except Exception:
                                pass
                            
                            # Log-Normal
                            try:
                                popt_log, _ = curve_fit(log_normal, bin_centers, hist, p0=[1, 1, 1], maxfev=5000)
                                y_log = log_normal(bin_centers, *popt_log)
                                r2_log = 1 - np.sum((hist - y_log)**2) / np.sum((hist - hist.mean())**2)
                                results['Log-Normal'] = {
                                    'params': popt_log,
                                    'r2': r2_log,
                                    'formula': f'F(x) = {popt_log[0]:.3f} * exp(-(log(x) - {popt_log[1]:.3f})² / (2 * {popt_log[2]:.3f}²))',
                                    'y_fit': y_log
                                }
                            except Exception:
                                pass
                            
                            # Gamma Distribution
                            try:
                                # Initial parameter estimates
                                mean_dist = np.mean(distances)
                                var_dist = np.var(distances)
                                # Method of moments for initial parameters
                                beta_init = var_dist / mean_dist  # scale parameter
                                alpha_init = mean_dist / beta_init  # shape parameter
                                
                                popt_gamma, _ = curve_fit(gamma_dist, bin_centers, hist, 
                                                        p0=[1, alpha_init, 1/beta_init], 
                                                        maxfev=5000,
                                                        bounds=([0.001, 0.001, 0.001], [np.inf, np.inf, np.inf]))
                                y_gamma = gamma_dist(bin_centers, *popt_gamma)
                                r2_gamma = 1 - np.sum((hist - y_gamma)**2) / np.sum((hist - hist.mean())**2)
                                results['Gamma'] = {
                                    'params': popt_gamma,
                                    'r2': r2_gamma,
                                    'formula': f'F(x) = {popt_gamma[0]:.3f} * (x^{popt_gamma[1]:.3f}) * exp(-{popt_gamma[2]:.3f} * x)',
                                    'y_fit': y_gamma
                                }
                            except Exception:
                                pass
                            
                            # Create plot
                            fig_dist_func = go.Figure()
                            
                            # Add histogram
                            fig_dist_func.add_trace(go.Bar(
                                x=bin_centers,
                                y=hist,
                                name='Observed',
                                opacity=0.6
                            ))
                            
                            # Add fitted curves
                            colors = ['red', 'blue', 'green']
                            for i, (name, result) in enumerate(results.items()):
                                fig_dist_func.add_trace(go.Scatter(
                                    x=bin_centers,
                                    y=result['y_fit'],
                                    mode='lines',
                                    name=f"{name} (R²={result['r2']:.3f})",
                                    line=dict(color=colors[i], width=2)
                                ))
                            
                            fig_dist_func.update_layout(
                                title=f"Trip Length Distribution Function - {purpose}",
                                xaxis_title="Distance (km)",
                                yaxis_title="Probability Density",
                                height=400,
                                xaxis=dict(range=[0, 100])
                            )
                            
                            st.plotly_chart(fig_dist_func, use_container_width=True)
                            
                            # Show parameters table
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Create a more elegant parameter table
                                st.write("**Model Fitting Results**")
                                
                                param_data = []
                                for name, result in results.items():
                                    param_data.append({
                                        'Distribution': name,
                                        'R²': f"{result['r2']:.4f}",
                                        'Quality': '★' * min(5, int(result['r2'] * 5 + 0.5))
                                    })
                                
                                param_df = pd.DataFrame(param_data)
                                st.dataframe(param_df, use_container_width=True, hide_index=True)
                                
                                # Show formulas in a collapsible section
                                with st.expander("View Mathematical Formulas"):
                                    for name, result in results.items():
                                        st.write(f"**{name}:**")
                                        st.code(result['formula'], language='text')
                            
                            with col2:
                                # Create a statistics table
                                st.write("**Distribution Statistics**")
                                
                                stats_data = {
                                    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Sample Size'],
                                    'Value': [
                                        f"{distances.mean():.2f} km",
                                        f"{np.median(distances):.2f} km",
                                        f"{distances.std():.2f} km",
                                        f"{distances.min():.2f} km",
                                        f"{distances.max():.2f} km",
                                        f"{len(distances):,} trips"
                                    ]
                                }
                                stats_df = pd.DataFrame(stats_data)
                                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
            else:
                st.info("No valid trip distance data available for the selected filters.")
        else:
            st.info("Trip coordinate columns not found in the dataset. Cannot calculate trip distances.")
    
    # Additional Insights for Trip Distribution Tab
    with tab2:
        st.markdown("---")
        with st.expander("📈 Trip Purpose Transition Analysis", expanded=False):
            try:
                # Analyze trip chains - what activities follow what
                if len(filtered_trips) > 0:
                    # Sort trips by person and trip number
                    trip_chains = filtered_trips.sort_values(['Person_id', 'Trip_Number'])
                    
                    # Create origin-destination purpose matrix
                    transitions = []
                    for person_id in trip_chains['Person_id'].unique()[:1000]:  # Limit to first 1000 persons for performance
                        person_trips = trip_chains[trip_chains['Person_id'] == person_id]
                        for i in range(len(person_trips) - 1):
                            origin_purpose = person_trips.iloc[i]['Trip_Destination_Purpose_Std']
                            next_purpose = person_trips.iloc[i + 1]['Trip_Destination_Purpose_Std']
                            transitions.append({
                                'From': origin_purpose,
                                'To': next_purpose,
                                'Count': 1
                            })
                    
                    if transitions:
                        transition_df = pd.DataFrame(transitions)
                        transition_matrix = transition_df.groupby(['From', 'To'])['Count'].sum().reset_index()
                        
                        # Create Sankey diagram
                        all_nodes = list(set(transition_matrix['From'].unique()) | set(transition_matrix['To'].unique()))
                        node_indices = {node: idx for idx, node in enumerate(all_nodes)}
                        
                        fig_sankey = go.Figure(data=[go.Sankey(
                            node=dict(
                                pad=15,
                                thickness=20,
                                line=dict(color="black", width=0.5),
                                label=all_nodes
                            ),
                            link=dict(
                                source=[node_indices[x] for x in transition_matrix['From']],
                                target=[node_indices[x] for x in transition_matrix['To']],
                                value=transition_matrix['Count']
                            )
                        )])
                        
                        fig_sankey.update_layout(
                            title="Trip Purpose Transitions (Trip Chains)",
                            height=600
                        )
                        
                        st.plotly_chart(fig_sankey, use_container_width=True)
                        
                        # Most common trip chains
                        st.subheader("Most Common Trip Purpose Sequences")
                        top_transitions = transition_matrix.nlargest(10, 'Count')
                        st.dataframe(
                            top_transitions.style.format({'Count': '{:,.0f}'}),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No trip chain data available for analysis.")
                else:
                    st.info("No trip data available for transition analysis.")
            except Exception as e:
                st.info("Unable to generate trip transition analysis. This may be due to insufficient data.")
    
    with tab3:
        st.session_state.active_tab = 'Level 3: Mode Choice'
        st.header("Mode Share Analysis")
        st.markdown("This analysis shows how people travel (transportation modes used).")
        
        # Default grouping for mode share
        mode_groupby = 'Status'
        
        # Calculate mode share
        mode_share = calculate_mode_share(filtered_trips, filtered_persons, [mode_groupby])
        # Stacked bar chart for mode share
        fig_mode = px.bar(
            mode_share,
            x=mode_groupby,
            y=f'percentage_by_{mode_groupby}',
            color='Mode_Clean',
            title=f"Mode Share by {mode_groupby}",
            labels={f'percentage_by_{mode_groupby}': 'Mode Share (%)', 'Mode_Clean': 'Travel Mode'},
            text=f'percentage_by_{mode_groupby}'
        )
        fig_mode.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        fig_mode.update_layout(barmode='stack', height=500)
        st.plotly_chart(fig_mode, use_container_width=True)
        
        # Mode share details
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall mode distribution
            overall_modes = mode_share.groupby('Mode_Clean')['trip_count'].sum().sort_values(ascending=False)
            
            fig_mode_pie = px.pie(
                values=overall_modes.values,
                names=overall_modes.index,
                title="Overall Mode Distribution",
                hole=0.4
            )
            fig_mode_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_mode_pie, use_container_width=True)
        
        with col2:
            # Top modes table
            st.subheader("Top Transportation Modes")
            top_modes = overall_modes.head(10).reset_index()
            top_modes.columns = ['Mode', 'Trip Count']
            top_modes['Percentage'] = (top_modes['Trip Count'] / top_modes['Trip Count'].sum() * 100).round(2)
            
            st.dataframe(
                top_modes.style.format({
                    'Trip Count': '{:,.0f}',
                    'Percentage': '{:.2f}%'
                }),
                use_container_width=True
            )
        
        # Detailed mode share table
        st.subheader("Mode Share by Market Segment (%)")
        
        pivot_mode = mode_share.pivot_table(
            values=f'percentage_by_{mode_groupby}', 
            index='Mode_Clean', 
            columns=mode_groupby, 
            fill_value=0
        )
        
        st.dataframe(
            pivot_mode.style.format("{:.1f}%").background_gradient(cmap='Blues', axis=1),
            use_container_width=True
        )
        
        # Mode Share by Purpose Analysis
        st.subheader("Mode Share by Trip Purpose")
        
        # Merge trips with person data and calculate mode share by purpose
        trips_for_mode_purpose = filtered_trips.merge(
            filtered_persons[['Person_id', 'Status']], 
            on='Person_id'
        )
        
        # Group by purpose and mode
        mode_by_purpose = trips_for_mode_purpose.groupby(['Trip_Destination_Purpose_Std', 'Mode_of_Travel_1']).size().reset_index(name='trip_count')
        
        # Calculate percentages within each purpose
        total_by_purpose = mode_by_purpose.groupby('Trip_Destination_Purpose_Std')['trip_count'].transform('sum')
        mode_by_purpose['percentage'] = (mode_by_purpose['trip_count'] / total_by_purpose * 100).round(2)
        
        # Create stacked bar chart
        fig_mode_purpose = px.bar(
            mode_by_purpose,
            x='Trip_Destination_Purpose_Std',
            y='percentage',
            color='Mode_of_Travel_1',
            title='Mode Share by Trip Purpose',
            labels={'Trip_Destination_Purpose_Std': 'Trip Purpose', 'percentage': 'Mode Share (%)', 'Mode_of_Travel_1': 'Travel Mode'},
            text='percentage',
            height=500
        )
        
        fig_mode_purpose.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        fig_mode_purpose.update_layout(
            barmode='stack',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig_mode_purpose, use_container_width=True)
        
        # Detailed table
        pivot_mode_purpose = mode_by_purpose.pivot_table(
            values='percentage', 
            index='Mode_of_Travel_1', 
            columns='Trip_Destination_Purpose_Std', 
            fill_value=0
        )
        
        st.write("**Mode Share Percentages by Trip Purpose**")
        st.dataframe(
            pivot_mode_purpose.style.format("{:.1f}%").background_gradient(cmap='Greens', axis=1),
            use_container_width=True
        )
        
    # Additional Insights for Mode Choice Tab
    with tab3:
        st.markdown("---")
        with st.expander("📈 Mode Choice Analysis by Demographics", expanded=False):
            try:
                # Analyze mode choice patterns by different demographic factors
                demographic_var = st.selectbox(
                    "Select demographic variable:",
                    ['Gender', 'Nationality_Group', 'Income_Level', 'Age_Group'],
                    key='demo_mode'
                )
                
                # Create age groups if Age column exists
                if demographic_var == 'Age_Group' and 'Age' in filtered_persons.columns:
                    filtered_persons['Age_Group'] = pd.cut(
                        filtered_persons['Age'],
                        bins=[0, 18, 30, 45, 60, 100],
                        labels=['<18', '18-30', '31-45', '46-60', '60+'],
                        duplicates='drop'
                    )
                
                if demographic_var in filtered_persons.columns or (demographic_var == 'Age_Group' and 'Age' in filtered_persons.columns):
                    # Merge for analysis
                    trips_demo = filtered_trips.merge(
                        filtered_persons[['Person_id', demographic_var]], 
                        on='Person_id'
                    )
                    
                    if len(trips_demo) > 0:
                        # Calculate mode share by demographic
                        mode_demo = trips_demo.groupby([demographic_var, 'Mode_of_Travel_1']).size().reset_index(name='count')
                        
                        # Calculate percentages
                        total_by_demo = mode_demo.groupby(demographic_var)['count'].transform('sum')
                        mode_demo['percentage'] = (mode_demo['count'] / total_by_demo * 100).round(2)
                        
                        # Create visualization
                        fig_demo_mode = px.bar(
                            mode_demo,
                            x=demographic_var,
                            y='percentage',
                            color='Mode_of_Travel_1',
                            title=f'Mode Share by {demographic_var}',
                            labels={'percentage': 'Mode Share (%)', 'Mode_of_Travel_1': 'Travel Mode'},
                            text='percentage',
                            height=500
                        )
                        
                        fig_demo_mode.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                        fig_demo_mode.update_layout(barmode='stack')
                        
                        st.plotly_chart(fig_demo_mode, use_container_width=True)
                        
                        # Statistical summary
                        with st.expander(f"Mode Preference Index by {demographic_var}"):
                            # Calculate preference index (ratio to overall average)
                            overall_mode_share = filtered_trips['Mode_of_Travel_1'].value_counts(normalize=True)
                            
                            preference_data = []
                            for demo in mode_demo[demographic_var].unique():
                                demo_data = mode_demo[mode_demo[demographic_var] == demo]
                                for _, row in demo_data.iterrows():
                                    mode = row['Mode_of_Travel_1']
                                    demo_share = row['percentage'] / 100
                                    overall_share = overall_mode_share.get(mode, 0.001)
                                    preference_index = demo_share / overall_share
                                    
                                    preference_data.append({
                                        demographic_var: demo,
                                        'Mode': mode,
                                        'Preference Index': preference_index
                                    })
                            
                            if preference_data:
                                preference_df = pd.DataFrame(preference_data)
                                pivot_preference = preference_df.pivot_table(
                                    values='Preference Index',
                                    index='Mode',
                                    columns=demographic_var,
                                    fill_value=0
                                )
                                
                                # Create heatmap
                                fig_pref = px.imshow(
                                    pivot_preference,
                                    labels=dict(x=demographic_var, y="Travel Mode", color="Preference Index"),
                                    title=f"Mode Preference Index by {demographic_var} (1.0 = average)",
                                    aspect="auto",
                                    color_continuous_scale='RdBu_r',
                                    color_continuous_midpoint=1.0
                                )
                                
                                st.plotly_chart(fig_pref, use_container_width=True)
                                
                                st.info("Preference Index: Values > 1.0 indicate higher than average preference, < 1.0 indicate lower preference")
                    else:
                        st.info("No trip data available for the selected demographic analysis.")
                else:
                    st.info(f"{demographic_var} data not available in the dataset.")
            except Exception as e:
                st.info("Unable to generate demographic analysis. Please try a different demographic variable.")

if __name__ == "__main__":
    main()