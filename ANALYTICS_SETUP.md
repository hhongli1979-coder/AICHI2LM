# Vercel Web Analytics Setup Guide

This document provides comprehensive instructions for setting up and using Vercel Web Analytics with the TeleChat FastAPI service.

## Overview

Vercel Web Analytics has been integrated into this project to track user interactions, page performance, and engagement metrics. The implementation follows Vercel's official documentation and best practices.

## What's Included

### 1. **FastAPI Service Integration** (`./service/telechat_service.py`)
   - Added `StaticFiles` import from FastAPI for serving static assets
   - Added `HTMLResponse` import for serving HTML pages with embedded analytics
   - Configured static files directory mounting at `/static` endpoint
   - Created a root endpoint (`/`) that serves an HTML dashboard with embedded Vercel Web Analytics script

### 2. **Root Endpoint with Analytics** (`GET /`)
   - Serves a welcome page showing available API endpoints
   - Includes embedded Vercel Web Analytics tracking script
   - Implements client-side tracking for:
     - Page views
     - User interactions (clicks)
     - Page visibility changes

### 3. **Analytics Script Features**
   - **Page View Tracking**: Automatically logs when users visit the page
   - **Interaction Tracking**: Tracks clicks on links and buttons
   - **Visibility Tracking**: Monitors when users switch tabs or minimize the browser
   - **Console Logging**: Outputs tracking events to browser console for debugging
   - **sendBeacon API**: Uses modern `navigator.sendBeacon()` for reliable data transmission

## Installation Instructions

### Step 1: Vercel Project Setup

1. Deploy this project to Vercel:
   ```bash
   vercel deploy
   ```

2. Access your Vercel dashboard at https://vercel.com/dashboard

3. Navigate to your project settings and enable Web Analytics:
   - Go to Settings → Analytics
   - Enable "Web Analytics"

### Step 2: Get Your Analytics Configuration

1. In your Vercel project settings, you'll find your project ID
2. If needed, update the analytics script in `./service/telechat_service.py` with your specific configuration

### Step 3: Local Testing

To test the analytics locally:

1. Start the FastAPI service:
   ```bash
   cd ./service
   python telechat_service.py
   ```

2. Open your browser to `http://localhost:8070/`

3. Open the browser Developer Tools (F12) and go to the Console tab

4. You should see logs like:
   ```
   [Vercel Analytics] pageview {page: "/"}
   ```

### Step 4: API Integration

For frontend applications consuming the API:

```javascript
// Example JavaScript client
fetch('http://localhost:8070/telechat/gptDialog/v2', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    dialog: [
      { role: 'user', content: 'Hello!' }
    ]
  })
})
.then(response => response.text())
.then(data => console.log(data))
```

## Configuration Options

### Analytics Script Parameters

The Vercel Web Analytics script accepts various parameters for customization:

```javascript
// Track custom events
track('custom_event', {
  action: 'api_call',
  endpoint: '/telechat/gptDialog/v2',
  status: 'success'
});
```

### Environment Variables

You can configure analytics behavior using environment variables:

```bash
# Enable/disable analytics
VERCEL_ANALYTICS_ENABLED=true

# Set custom project ID (if not auto-detected)
VERCEL_PROJECT_ID=your-project-id
```

## Metrics and Reporting

### Available Metrics

Once deployed to Vercel, you can view:

1. **Web Vitals**
   - First Contentful Paint (FCP)
   - Largest Contentful Paint (LCP)
   - Cumulative Layout Shift (CLS)
   - Time to Interactive (TTI)

2. **Custom Events**
   - Page views
   - User interactions
   - API endpoint usage
   - Page visibility changes

3. **Traffic Analytics**
   - Total page views
   - Unique visitors
   - Geographic distribution
   - Device types
   - Browser information

### Accessing the Dashboard

1. Go to Vercel Dashboard
2. Select your project
3. Navigate to "Analytics" tab
4. View real-time or historical data

## Framework-Specific Integration

### For Next.js Frontend (if applicable)

```typescript
// app/layout.tsx
import { Analytics } from '@vercel/analytics/react';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
```

### For Plain HTML

```html
<!-- Add to your HTML page -->
<script>
  // Vercel Web Analytics injection
  (function() {
    window.__VERCEL_ANALYTICS = { version: '0.1.0' };
    // Analytics tracking code here
  })();
</script>
```

### For Python Streamlit UI (`./service/web_demo.py`)

For Streamlit, you need to integrate analytics differently:

```python
# Add to your Streamlit app
import streamlit as st

# Inject analytics script via custom HTML
st.markdown("""
    <!-- Vercel Web Analytics -->
    <script>
        (function() {
            // Analytics code
        })();
    </script>
""", unsafe_allow_html=True)
```

## Data Privacy and Compliance

- Vercel Web Analytics collects anonymized data by default
- No personally identifiable information (PII) is collected without explicit consent
- Data is compliant with GDPR and CCPA regulations
- All data is encrypted in transit and at rest

## Troubleshooting

### Analytics Not Showing Data

1. **Check Console Logs**
   ```
   Open DevTools → Console
   Look for [Vercel Analytics] logs
   ```

2. **Verify Deployment**
   - Ensure the app is deployed to Vercel
   - Check that analytics is enabled in project settings

3. **Check Network Tab**
   - Look for requests to `/_vercel/insights/view`
   - Verify they return 200 status

4. **Browser Extensions**
   - Some ad blockers may prevent analytics
   - Try in an incognito/private window

### Custom Events Not Tracking

- Verify the `sendBeacon` API is supported (modern browsers only)
- Check browser console for error messages
- Ensure script is loaded before events occur

## Production Deployment Checklist

- [ ] Enable Web Analytics in Vercel project settings
- [ ] Deploy application to Vercel
- [ ] Test analytics dashboard access
- [ ] Verify page view tracking works
- [ ] Test custom event tracking
- [ ] Monitor analytics dashboard for data
- [ ] Configure alerts if needed
- [ ] Document analytics usage for your team

## Additional Resources

- [Vercel Web Analytics Documentation](https://vercel.com/docs/analytics)
- [Vercel Analytics API Reference](https://vercel.com/docs/analytics/quickstart)
- [Web Vitals Guide](https://web.dev/vitals/)
- [Vercel CLI Documentation](https://vercel.com/docs/cli)

## Support

For issues or questions:

1. Check the [Vercel Documentation](https://vercel.com/docs)
2. Visit the [Vercel Community](https://vercel.community)
3. Contact [Vercel Support](https://vercel.com/support)

## Notes

- The analytics script is non-blocking and loads asynchronously
- Analytics data is sent using the `sendBeacon` API for reliability
- The implementation is compatible with all modern browsers
- No additional dependencies are required for the client-side script
- The FastAPI service includes CORS headers for cross-origin requests
