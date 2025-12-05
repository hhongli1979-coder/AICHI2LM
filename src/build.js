#!/usr/bin/env node

/**
 * Build script for AICHI-2-LM frontend
 * 
 * This script prepares the application for production deployment.
 * In a real setup, this would bundle the JavaScript and prepare assets.
 */

import { fileURLToPath } from 'url';
import { dirname } from 'path';
import fs from 'fs';
import path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('🔨 Building AICHI-2-LM application...');

const buildDir = path.join(__dirname, '..', 'dist');
const srcDir = path.join(__dirname, '..');

// Create dist directory if it doesn't exist
if (!fs.existsSync(buildDir)) {
  fs.mkdirSync(buildDir, { recursive: true });
}

// Copy public files to dist
const publicDir = path.join(srcDir, 'public');
if (fs.existsSync(publicDir)) {
  if (!fs.existsSync(path.join(buildDir, 'public'))) {
    fs.mkdirSync(path.join(buildDir, 'public'), { recursive: true });
  }
  
  fs.readdirSync(publicDir).forEach(file => {
    const src = path.join(publicDir, file);
    const dest = path.join(buildDir, 'public', file);
    fs.copyFileSync(src, dest);
  });
  
  console.log('✅ Public assets copied');
}

// Copy src files to dist
if (!fs.existsSync(path.join(buildDir, 'src'))) {
  fs.mkdirSync(path.join(buildDir, 'src'), { recursive: true });
}

fs.readdirSync(path.join(srcDir, 'src')).forEach(file => {
  if (file.endsWith('.js')) {
    const src = path.join(srcDir, 'src', file);
    const dest = path.join(buildDir, 'src', file);
    fs.copyFileSync(src, dest);
  }
});

console.log('✅ Source files processed');

// Create a simple build manifest
const manifest = {
  version: '1.0.0',
  buildDate: new Date().toISOString(),
  analytics: 'vercel',
  entries: {
    html: 'public/index.html',
    app: 'src/app.js',
    analytics: 'src/analytics.js'
  }
};

fs.writeFileSync(
  path.join(buildDir, 'manifest.json'),
  JSON.stringify(manifest, null, 2)
);

console.log('✅ Build manifest created');

console.log('');
console.log('🎉 Build completed successfully!');
console.log(`📁 Output directory: ${buildDir}`);
console.log('');
