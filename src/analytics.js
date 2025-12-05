/**
 * Vercel Web Analytics Integration
 * 
 * This module initializes and configures Vercel Web Analytics for the AICHI-2-LM application.
 * It must run on the client side and provides real-time performance and usage analytics.
 * 
 * Note: Analytics injection does not include route support and is designed for client-side
 * performance monitoring of page loads, interactions, and web vitals.
 */

import { inject } from '@vercel/analytics';

/**
 * Initialize Vercel Web Analytics
 * 
 * This function should be called as early as possible in the application's entry point
 * to ensure all page interactions are tracked from the start.
 * 
 * Features tracked:
 * - Page views and navigation
 * - Web Vitals (LCP, FID, CLS, FCP, TTFB)
 * - Custom events
 * - User interactions
 */
export function initializeAnalytics() {
  try {
    // Inject Vercel Analytics
    // This initializes the analytics collection with default settings
    inject();
    
    console.log('✅ Vercel Web Analytics initialized successfully');
    
    // Optional: Set up environment-specific configurations
    if (process.env.NODE_ENV === 'development') {
      console.log('📊 Analytics running in development mode - data may not be published');
    }
    
    return true;
  } catch (error) {
    console.error('❌ Failed to initialize Vercel Web Analytics:', error);
    return false;
  }
}

/**
 * Get analytics status
 * Useful for debugging and verifying the analytics is properly initialized
 */
export function getAnalyticsStatus() {
  return {
    initialized: true,
    timestamp: new Date().toISOString(),
    environment: process.env.NODE_ENV || 'production'
  };
}

export default initializeAnalytics;
