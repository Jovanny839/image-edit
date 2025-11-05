# AI Image Editor API

A FastAPI-based service for editing images using OpenAI's image editing capabilities and comparing character similarity using CLIP embeddings.

## Features

- **Image Editing**: Edit images using OpenAI's image editing API
- **Image Storage**: Upload edited images to Supabase storage
- **Character Similarity**: Compare cartoon character images using CLIP embeddings
- **RESTful API**: Well-documented API with OpenAPI/Swagger documentation

## Tech Stack

- FastAPI
- OpenAI API
- Supabase (for storage)
- CLIP (for character similarity)
- PyTorch
- Pillow

## Deployment on Render.com

This project is configured for deployment on Render.com.

### Prerequisites

1. A Render.com account
2. A GitHub repository with this code
3. OpenAI API key
4. Supabase account (optional, for image storage)

### Deployment Steps

1. **Connect Repository to Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select the repository and branch

2. **Configure Environment Variables**
   In the Render dashboard, add these environment variables:
   - `OPENAI_API_KEY` - Your OpenAI API key (required)
   - `SUPABASE_URL` - Your Supabase project URL (optional)
   - `SUPABASE_ANON_KEY` - Your Supabase anonymous key (optional)
   - `SUPABASE_SERVICE_KEY` - Your Supabase service role key (optional, for storage operations)

3. **Deploy**
   - Render will automatically detect the `render.yaml` file
   - The service will build and deploy automatically
   - Build command: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`

4. **Verify Deployment**
   - Health check: `https://your-service.onrender.com/health`
   - API docs: `https://your-service.onrender.com/docs`

### Configuration Files

- `render.yaml` - Render.com service configuration
- `runtime.txt` - Python version specification (3.11.9)
- `requirements.txt` - Python dependencies

### API Endpoints

- `GET /` - API information
- `GET /health` - Health check endpoint
- `GET /docs` - OpenAPI/Swagger documentation
- `POST /edit-image/` - Edit an image from URL
- `POST /edit-image-stream/` - Edit an image and return as stream
- `POST /compare-similarity/` - Compare similarity between two character images

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your environment variables:
   ```
   OPENAI_API_KEY=your_key_here
   SUPABASE_URL=your_url_here
   SUPABASE_ANON_KEY=your_key_here
   SUPABASE_SERVICE_KEY=your_key_here
   ```

3. Run the server:
   ```bash
   python main.py
   ```

4. Access the API:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

### Notes

- The service uses Python 3.11.9
- Render automatically sets the `PORT` environment variable
- The health check endpoint is configured at `/health`
- The service runs with 1 worker by default (suitable for starter plan)