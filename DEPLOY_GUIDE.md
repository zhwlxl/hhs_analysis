# Deployment Guide

## Option 1: Deploy to Streamlit Cloud (Recommended)

1. **Create GitHub Repository**
   ```bash
   cd deploy
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/hhs-dashboard.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Get your app URL**
   - It will be something like: `https://hhs-dashboard-xxxxx.streamlit.app`

## Option 2: Deploy Static Site to Netlify

1. **Update the iframe URL in index.html**
   - Open `index.html`
   - Find line 134: `src="https://your-hhs-dashboard.streamlit.app"`
   - Replace with your Streamlit Cloud URL

2. **Deploy to Netlify**
   - Go to [netlify.com](https://netlify.com)
   - Drag and drop the `deploy` folder
   - Your static site will be live with an embedded dashboard

## Option 3: Direct Netlify Deploy (Static Only)

If you only want a landing page on Netlify:

1. Create `netlify.toml`:
   ```toml
   [[redirects]]
     from = "/*"
     to = "/index.html"
     status = 200
   ```

2. Deploy only `index.html` and `netlify.toml` to Netlify

## Local Testing

```bash
cd deploy
pip install -r requirements.txt
streamlit run app.py
```

Then open `index.html` in a browser - it will automatically connect to localhost:8501