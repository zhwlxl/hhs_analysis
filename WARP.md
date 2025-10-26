# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Streamlit-based dashboard for analyzing Household Travel Survey (HHS) data from Saudi Arabia. Provides four levels of analysis:
- **Level 1**: Mobility Rates (trip and tour rates by market segments)
- **Level 2**: Trip Distribution (destinations and activities)
- **Level 3**: Mode Choice (transportation mode analysis with logit modeling)
- **Level 4**: Time Choice (when people travel - time-of-day activity profiles)

## Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# App runs on http://localhost:8501 by default
```

### Deployment
The app is designed for Streamlit Cloud deployment:
1. Push to GitHub repository
2. Deploy via [share.streamlit.io](https://share.streamlit.io)
3. Select `app.py` as main file

See `DEPLOY_GUIDE.md` for detailed deployment instructions.

## Architecture

### Data Pipeline
1. **Data Loading** (`load_and_process_data()` in `app.py`):
   - Reads CSV files from `csv_data_optimized/` directory
   - Three main datasets: HHS.csv (households), Persons.csv, Trips.csv
   - Uses `@st.cache_data` decorator for performance

2. **Data Processing**:
   - **Status Categorization**: Maps occupation types to status codes (W=Worker, S=Student, SS=Secondary Student, SP=Primary Student, NW=Non-Worker)
   - **Nationality Grouping**: Categorizes into Saudi Arab, Non-Saudi Arab, Non-Saudi Non-Arab
   - **Trip Purpose Standardization**: Maps detailed purposes to simplified categories (Home, Work, School, Business, Shopping, Drop-off, Others)
   - **Tour Calculation**: Identifies complete round trips from home using trip origin/destination purposes
   - **Car Availability**: Determined by license type and household vehicle ownership

3. **Analysis Modules**:
   - **Mobility Rates**: `calculate_mobility_rates()` - calculates trip/tour rates by market segments
   - **Trip Distribution**: `calculate_trip_distribution()` - analyzes destination purposes by demographics
   - **Mode Choice**: Two implementations available:
     - `mode_choice_model.py`: Full multinomial logit with scipy optimization
     - `mode_choice_model_simple.py`: Simplified version with preset parameters (used in production)
   - **Time Choice**: Analyzes time-of-day travel patterns by activity purpose and market segments

### Mode Choice Model Architecture

The mode choice model uses multinomial logit framework:

- **Mode Categories**: Private Vehicle, Walk, Public Transit, Taxi/Rideshare
- **Mode Mapping**: Maps detailed survey modes to main categories (e.g., "Car Driver" → "Private Vehicle")
- **Attributes Calculated**:
  - Cost per trip (fuel, parking, fares based on Saudi Arabia context)
  - Travel time (using distance and mode-specific speeds)
  - Mode availability (based on license, trip distance constraints)
- **Haversine Distance**: Calculates trip distance from lat/lon coordinates
- **Cost Parameters**: Configurable assumptions for fuel prices, fares, speeds (see `_initialize_cost_parameters()`)

### Key Data Structures

**Market Segmentation Variables**:
- `Status`: W, NW, S, SS, SP (from occupation)
- `Gender`: Male, Female
- `Nationality_Group`: Saudi Arab, Non-Saudi Arab, Non-Saudi Non-Arab
- `Car_Available`: CA (Car Available), NCA (No Car Available)
- `Income_Level`: 3H (High), 2M (Medium), 1L (Low) - synthetic proxy based on occupation

**Trip Data Fields**:
- `Trip_Origin_Purpose` / `Trip_Destination_Purpose`: Raw survey purposes
- `Trip_Origin_Purpose_Std` / `Trip_Destination_Purpose_Std`: Standardized to 7 categories
- `Mode_of_Travel_1`: Primary travel mode
- `Trip_Departure_Time` / `Trip_Arrival_Time`: Time data in HH:MM:SS format for temporal analysis
- `Origin_Latitude`, `Origin_Longitude`, `Destination_Latitude`, `Destination_Longitude`: For distance calculation

### UI Organization

- **Sidebar**: Filtering controls (sample size, status, gender, nationality, car availability, income)
- **Main Panel**: Four tabbed analysis levels with interactive Plotly visualizations
- **Streamlit Configuration**: Custom CSS for compact layout (`.streamlit/config.toml`)

### Time Choice Analysis (Level 4)

Analyzes when people travel throughout the day:
- **Overall Time Profile**: Hourly distribution of all trips (bar chart)
- **Time by Activity**: Line charts showing departure times for different activities (Work, School, Shopping, etc.)
- **Activity Heatmap**: When different activities occur throughout the day (percentage-based)
- **Peak Hours Analysis**: Identifies peak travel times for each activity and time period distribution
- **Time by Market Segment**: Compare travel timing patterns across Status, Gender, Car Availability
- **Detailed Cross-Analysis**: Interactive filtering by activity + market segment with statistics

## Important Implementation Notes

- **Caching**: Use `@st.cache_data` for data loading functions to avoid reprocessing on every interaction
- **Tour Definition**: A tour = one "home → out" trip paired with one "out → home" trip (minimum of both counts)
- **Mode Availability Logic**: 
  - Private Vehicle requires license AND distance > 0.5 km
  - Walk only available for trips ≤ 5 km
  - Public Transit for trips > 1 km
  - Taxi always available
- **Occupation Status Mapping**: Uses `OCCUPATION_TO_STATUS` dict and `categorize_status()` function with keyword matching for student types
- **Activity Mapping**: Uses `ACTIVITY_MAPPING` dict with fallback partial matching in `standardize_purpose()`

## Data Location

All survey data is in `csv_data_optimized/`:
- `HHS.csv`: Household-level data
- `Persons.csv`: Individual person records with demographics
- `Trips.csv`: Trip-level records with origin/destination/mode/time

**Note**: Time columns (`Trip_Departure_Time`, `Trip_Arrival_Time`) are merged from the original `../csv_data/Jaddah_Trips.csv` file using the `add_time_columns.py` script.

## Testing

No automated tests currently in place. Manual testing via:
1. Run `streamlit run app.py`
2. Test filter combinations in sidebar
3. Verify visualizations render correctly for each analysis level
4. Check mode choice predictions with policy scenarios

## Python Version

Runtime specified in `runtime.txt` (Python version for Streamlit Cloud deployment)
