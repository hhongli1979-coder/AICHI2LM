/**
 * Main Application Entry Point
 * 
 * This is the primary entry point for the AICHI-2-LM frontend application.
 * It initializes all critical services including Vercel Web Analytics.
 */

import initializeAnalytics from './analytics.js';

/**
 * Initialize the application
 * Called when the page loads to set up all necessary services
 */
async function initializeApp() {
  console.log('🚀 Initializing AICHI-2-LM application...');
  
  // Initialize Vercel Web Analytics first
  // This must happen on the client side
  const analyticsInitialized = initializeAnalytics();
  
  if (analyticsInitialized) {
    console.log('📊 Analytics tracking is active');
  }
  
  console.log('✅ Application initialized successfully');
}

// Run initialization when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp);
} else {
  initializeApp();
}

export { initializeApp };
