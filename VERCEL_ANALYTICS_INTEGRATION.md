# Vercel Web Analytics Integration Guide

## Overview

Vercel Web Analytics has been successfully integrated into the TeleChat API project. This document describes the implementation and how to use it.

## What is Vercel Web Analytics?

Vercel Web Analytics is a privacy-first analytics solution that tracks Web Vitals and provides insights into your application's performance and user experience. It requires no cookies and is GDPR compliant.

## Implementation Details

### Components Added

#### 1. **Static HTML Landing Page** (`service/static/index.html`)
- A welcome page served at the root endpoint (`/`) of the FastAPI application
- Includes the Vercel Web Analytics script tag: `/_vercel/insights/script.js`
- Provides links to API documentation and other resources
- Responsive design with a modern UI

#### 2. **FastAPI Service Updates** (`service/telechat_service.py`)
- **Static Files Mounting**: Added `StaticFiles` configuration to serve static assets
- **Root Endpoint**: New `GET /` endpoint that serves the analytics-enabled HTML page
- **Imports**: Added necessary imports for `FileResponse`, `StaticFiles`, and `Path`

### How It Works

The integration uses Vercel's Web Analytics script in plain HTML format (no npm package installation needed for Python projects):

```html
<script defer src="/_vercel/insights/script.js"></script>
```

This script:
1. Automatically tracks Core Web Vitals (LCP, FID, CLS, FCP, TTFB)
2. Sends analytics data to Vercel's edge network
3. Requires no configuration or initialization code
4. Works client-side in browsers that visit the application

### File Structure

```
service/
├── telechat_service.py       # Updated FastAPI app with analytics support
├── static/
│   └── index.html           # Welcome page with analytics script
├── web_demo.py              # Streamlit web interface
└── [other service files]
```

## Deployment Instructions

### Prerequisites
- Python 3.7+
- FastAPI (already in requirements.txt)
- Uvicorn (already in requirements.txt)

### Installation

No additional package installation is required! The Vercel Web Analytics script is loaded directly from Vercel's CDN.

### Running the Application

The application can be started as usual:

```bash
# Using the deployment script
python deploy.py

# Or directly running the service
cd service
python telechat_service.py
```

### Accessing the Application

Once running, visit the API in your browser:

- **Root Page (with Analytics)**: `http://localhost:8070/`
- **API Documentation**: `http://localhost:8070/docs`
- **Alternative Docs**: `http://localhost:8070/redoc`

## Vercel Integration

To fully utilize Vercel Web Analytics:

### 1. Deploy to Vercel

If deploying on Vercel, the analytics will automatically collect data. Here's a basic `vercel.json` configuration:

```json
{
  "buildCommand": "pip install -r requirements.txt",
  "env": {
    "PYTHONUNBUFFERED": "1"
  }
}
```

### 2. Enable Analytics in Vercel Dashboard

1. Go to your Vercel project dashboard
2. Navigate to **Settings → Analytics**
3. Enable **Web Analytics** if not already enabled
4. Your data will start appearing in the Vercel dashboard

### 3. View Analytics Data

Analytics data appears in the Vercel dashboard showing:
- Core Web Vitals (LCP, FID, CLS, FCP, TTFB)
- Page load performance
- User interaction metrics
- Geographic distribution of users
- Device and browser information

## Metrics Tracked

### Core Web Vitals

1. **LCP (Largest Contentful Paint)**
   - Measures loading performance
   - Marks the time when the main content has loaded

2. **FID (First Input Delay)**
   - Measures interactivity
   - Tracks the delay between user input and response

3. **CLS (Cumulative Layout Shift)**
   - Measures visual stability
   - Tracks unexpected layout shifts

### Additional Metrics

- **FCP (First Contentful Paint)**: When first content appears
- **TTFB (Time to First Byte)**: Server response time
- **Navigation Timing**: Page transitions
- **Resource Timing**: Asset loading performance

## Architecture

```
┌─────────────────────────────────────────────┐
│         User's Browser                       │
│  ┌─────────────────────────────────────────┐ │
│  │  HTML Page with Analytics Script        │ │
│  │  <script src="/_vercel/insights/...">   │ │
│  └─────────────────────────────────────────┘ │
└────────────────┬────────────────────────────┘
                 │ Collects Web Vitals
                 │ Sends metrics
                 ▼
      ┌──────────────────────────┐
      │  Vercel Edge Network     │
      │  /_vercel/insights/...   │
      └────────┬─────────────────┘
               │
               ▼
      ┌──────────────────────────┐
      │  Vercel Analytics        │
      │  Dashboard & Reporting   │
      └──────────────────────────┘
```

## Privacy & Compliance

- **No Cookies**: Vercel Web Analytics doesn't use cookies
- **GDPR Compliant**: No user identification or personal data collection
- **Privacy-First**: Only collects performance metrics
- **Anonymous**: All data is aggregated and anonymized

## Customization

### Adding Analytics to Other Pages

To add analytics to other HTML files served from FastAPI:

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <script defer src="/_vercel/insights/script.js"></script>
    <!-- Your other head content -->
</head>
<body>
    <!-- Your page content -->
</body>
</html>
```

### Programmatic Tracking (JavaScript)

When served as HTML, you can add custom events:

```html
<script>
// Track custom events (if using Vercel Web Analytics SDK)
window.va?.track('custom_event', {
    value: 42
});
</script>
```

### For Streamlit Frontend

The Streamlit web interface (`web_demo.py`) runs independently on port 8501. To add analytics to Streamlit:

1. Create a custom `_components/` folder with an analytics component
2. Or inject a script through Streamlit's custom HTML features
3. Reference: https://docs.streamlit.io/library/develop/custom-components

## Monitoring & Debugging

### Check if Analytics is Working

1. Open your browser's **Developer Tools** (F12)
2. Go to the **Network** tab
3. Look for requests to `/_vercel/insights/...`
4. Should see successful requests when pages load

### Common Issues

**Issue**: Analytics script not loading
- **Solution**: Ensure `static/` directory exists in `service/` folder
- **Solution**: Check that FastAPI static files are properly mounted

**Issue**: No data appearing in Vercel dashboard
- **Solution**: Ensure application is deployed to Vercel platform
- **Solution**: Wait 5-10 minutes for initial data to appear

## Best Practices

1. **Keep the Script**: The `/_vercel/insights/script.js` should remain in all served HTML
2. **Monitor Regularly**: Check the Vercel dashboard weekly for performance trends
3. **Set Baselines**: Establish performance baselines for comparison
4. **Act on Insights**: Use the data to optimize performance bottlenecks
5. **Update Content**: Refresh content based on user behavior patterns shown in analytics

## Testing

To verify the integration works:

```bash
# Start the service
cd service
python telechat_service.py

# In another terminal, test the root endpoint
curl -I http://localhost:8070/
# Should return 200 OK with HTML content-type

# Open in browser
# http://localhost:8070/
# Check browser network tab for analytics requests
```

## Troubleshooting

### Static Files Not Serving

If you see 404 errors for static files:

```python
# Verify static directory exists:
ls -la service/static/

# Verify index.html exists:
ls -la service/static/index.html
```

### Analytics Not Tracking

1. Check browser console for errors
2. Verify script URL: `/_vercel/insights/script.js`
3. Ensure page is actually being loaded (not API endpoint)
4. Check Vercel project settings

## References

- [Vercel Web Analytics Documentation](https://vercel.com/docs/product/analytics)
- [Web Vitals Guide](https://web.dev/vitals/)
- [FastAPI Static Files](https://fastapi.tiangolo.com/tutorial/static-files/)
- [Privacy Policy](https://vercel.com/legal/privacy-policy)

## Support

For issues or questions:

1. Check Vercel Documentation: https://vercel.com/docs
2. Review FastAPI Documentation: https://fastapi.tiangolo.com
3. Check browser console for JavaScript errors
4. Review service logs for FastAPI errors

## Summary

The Vercel Web Analytics integration is now complete and provides:

✅ **Zero-Configuration Analytics** - No setup needed beyond script inclusion
✅ **Privacy-First Approach** - GDPR compliant with no cookies
✅ **Core Web Vitals Tracking** - Automatic performance monitoring
✅ **Easy Deployment** - Works seamlessly when deployed to Vercel
✅ **Beautiful UI** - Professional landing page with analytics information

The analytics script will automatically track visitor behavior and performance metrics when the application is accessed through a browser.
