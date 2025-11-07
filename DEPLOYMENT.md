# Deployment Guide: Netlify + GitHub

This guide will help you deploy the Mammographic Cancer Stage Detection web application to Netlify using GitHub.

## Prerequisites

1. A GitHub account
2. A Netlify account (free tier is sufficient)
3. Your code pushed to a GitHub repository

## Important Notes

⚠️ **Netlify Limitations:**
- Netlify Functions have a **10-second timeout** on the free tier (26 seconds on paid)
- Memory limit of **1.5GB** per function
- Large ML models (like TensorFlow) may not fit or may timeout

**Current Setup:**
- The Netlify function uses a simplified version that extracts radiomics features
- Full U-Net and stage classifier models are not included due to size/timeout constraints
- For production use with full models, consider deploying the Flask backend separately (see Alternative Deployment section)

## Step 1: Prepare Your Repository

1. **Ensure all files are committed:**
   ```bash
   git add .
   git commit -m "Prepare for Netlify deployment"
   git push origin main
   ```

2. **Verify your repository structure includes:**
   - `public/` folder with `index.html`, `css/`, `js/`
   - `netlify.toml` configuration file
   - `netlify/functions/process_image.py`
   - `netlify/functions/requirements.txt`
   - `utils/radiomics_utils.py`

## Step 2: Connect GitHub to Netlify

1. **Log in to Netlify:**
   - Go to [https://app.netlify.com](https://app.netlify.com)
   - Sign in with your GitHub account

2. **Create a new site:**
   - Click "Add new site" → "Import an existing project"
   - Choose "GitHub" as your Git provider
   - Authorize Netlify to access your repositories if prompted

3. **Select your repository:**
   - Find and select your `Mammographic_Cancer_Stage_Detection` repository
   - Click "Connect"

## Step 3: Configure Build Settings

Netlify should auto-detect settings from `netlify.toml`, but verify:

- **Build command:** `echo 'No build step needed for static site'`
- **Publish directory:** `public`
- **Base directory:** (leave empty, or set to root)

**Note:** The build command is minimal since we're serving static files. The functions are deployed automatically.

## Step 4: Deploy

1. Click **"Deploy site"**
2. Wait for the deployment to complete (usually 1-2 minutes)
3. Your site will be live at a URL like: `https://your-site-name.netlify.app`

## Step 5: Test Your Deployment

1. Visit your deployed site
2. Upload a test mammogram image
3. Verify that:
   - The image uploads successfully
   - Radiomics features are extracted
   - Results are displayed (may be in demo mode)

## Troubleshooting

### Function Timeout Errors
If you see timeout errors:
- The function may be taking too long (>10 seconds)
- Consider optimizing image processing
- Or use the alternative deployment method (see below)

### Import Errors
If Python imports fail:
- Check that `netlify/functions/requirements.txt` includes all dependencies
- Ensure `utils/radiomics_utils.py` is accessible from the function
- Check Netlify build logs for specific error messages

### Large File Upload Issues
- Netlify has a 6MB request body limit on free tier
- Consider compressing images client-side before upload
- Or use a separate file storage service (S3, Cloudinary, etc.)

## Alternative Deployment: Separate Backend

For production use with full models, consider this architecture:

### Option A: Render/Railway for Backend + Netlify for Frontend

1. **Deploy Flask backend to Render/Railway:**
   - Use `app.py` (Flask version)
   - Set environment variables
   - Get backend URL (e.g., `https://your-backend.onrender.com`)

2. **Update frontend to use backend URL:**
   - Modify `public/js/main.js` to point to your backend
   - Change fetch URL from `/api/process_image` to `https://your-backend.onrender.com/upload`

3. **Deploy frontend to Netlify:**
   - Frontend remains on Netlify
   - Backend handles all processing

### Option B: Full Stack on Render/Railway

Deploy the entire Flask app (including templates) to:
- **Render:** [https://render.com](https://render.com)
- **Railway:** [https://railway.app](https://railway.app)
- **Heroku:** [https://heroku.com](https://heroku.com)

These platforms support:
- Longer timeouts (up to 30 minutes on Render)
- More memory
- Better Python/ML model support

## Environment Variables (if needed)

If you need to set environment variables in Netlify:

1. Go to Site settings → Environment variables
2. Add variables like:
   - `PYTHON_VERSION=3.11`
   - Any API keys or configuration

## Updating Your Site

After making changes:

1. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update site"
   git push origin main
   ```

2. Netlify will automatically rebuild and deploy (if auto-deploy is enabled)

## Support

For issues specific to:
- **Netlify:** Check [Netlify Docs](https://docs.netlify.com/)
- **Functions:** See [Netlify Functions Docs](https://docs.netlify.com/functions/overview/)
- **This project:** Check the main README.md

