# Demo Household Travel Survey Analysis Dashboard

This is a comprehensive dashboard for analyzing household travel survey data with three levels of analysis:
- **Level 1**: Mobility Rates (trip and tour rates)
- **Level 2**: Trip Distribution (destinations and activities)
- **Level 3**: Mode Choice (transportation modes)

## Deployment Options

### Option 1: Streamlit Cloud (Recommended)
1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your GitHub repository
4. Select `app.py` as the main file

### Option 2: Netlify with Static HTML
Since Netlify doesn't support Python apps directly, create an HTML redirect:
1. Deploy the `index.html` file to Netlify
2. Update the iframe URL with your Streamlit Cloud URL

### Option 3: Local Deployment
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Interactive filters for market segments, gender, nationality, etc.
- Real-time visualization updates
- Trip length distribution analysis
- Mode share analysis
- Cross-tabulation insights