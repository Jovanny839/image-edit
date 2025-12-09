# Google Vision API Setup Guide (Using Google AI Studio)

This guide explains how to set up Google Vision API for image quality validation using Google AI Studio (aistudio.google.com).

## Overview

The quality validation feature uses **Google Gemini Vision API** (via Google AI Studio) to analyze images for:
- Image quality and clarity
- Content appropriateness (especially for children's content)
- Technical issues (blur, distortion, artifacts)
- Resolution and detail sufficiency
- Overall visual quality assessment

## Setup Instructions

### Step 1: Get API Key from Google AI Studio

1. **Visit Google AI Studio**
   - Go to [https://aistudio.google.com](https://aistudio.google.com)
   - Sign in with your Google account

2. **Create or Get API Key**
   - Click on "Get API Key" in the left sidebar
   - If you don't have a project, create a new Google Cloud project
   - Copy your API key (it will look like: `AIzaSy...`)

3. **Enable Required APIs**
   - The Gemini API should be enabled automatically
   - If needed, you can enable it in [Google Cloud Console](https://console.cloud.google.com/apis/library)
   - Search for "Generative Language API" and ensure it's enabled

### Step 2: Configure Environment Variables

Add your API key to your `.env` file in the backend directory:

```env
GEMINI_API_KEY=your_api_key_here
```

**Note:** The same `GEMINI_API_KEY` used for image generation is also used for quality validation. No separate key is needed.

### Step 3: Verify Setup

1. Start your backend server:
   ```bash
   cd backend
   python main.py
   ```

2. Check the health endpoint:
   ```bash
   curl http://localhost:8000/health
   ```

   You should see:
   ```json
   {
     "quality_validation_enabled": true,
     "gemini_client_initialized": true
   }
   ```

## API Endpoints

### 1. Validate Image Quality (Standalone)

**Endpoint:** `POST /validate-image-quality/`

**Request:**
```json
{
  "image_url": "https://example.com/image.jpg",
  "prompt": "optional prompt (not used for validation)"
}
```

**Response:**
```json
{
  "success": true,
  "validation": {
    "is_valid": true,
    "quality_score": 0.85,
    "is_appropriate": true,
    "is_clear": true,
    "has_sufficient_detail": true,
    "issues": [],
    "recommendations": ["Image quality is good"],
    "details": {
      "image_properties": {
        "actual_resolution": "1024x768",
        "format": "JPEG",
        "clarity": "high",
        "brightness": "normal",
        "composition": "good"
      },
      "validation_available": true,
      "model_used": "gemini-1.5-pro"
    }
  }
}
```

### 2. Edit Image with Quality Validation

**Endpoint:** `POST /edit-image/`

The quality validation is automatically performed before image editing. The response includes validation results:

```json
{
  "success": true,
  "message": "Image edited and uploaded successfully",
  "storage_info": {...},
  "quality_validation": {
    "is_valid": true,
    "quality_score": 0.85,
    ...
  }
}
```

## Validation Criteria

The validation checks for:

1. **Quality Score** (0.0 - 1.0)
   - Overall image quality assessment
   - Threshold: ≥ 0.5 for valid images

2. **Appropriateness**
   - Content suitable for target audience
   - No inappropriate content

3. **Clarity**
   - Image is sharp and clear
   - No excessive blur or distortion

4. **Detail Sufficiency**
   - Image has sufficient detail for processing
   - Resolution is adequate

5. **Technical Issues**
   - Blur, artifacts, compression issues
   - Format compatibility

## Usage Examples

### Python Example

```python
import requests

# Validate image quality
response = requests.post(
    "http://localhost:8000/validate-image-quality/",
    json={
        "image_url": "https://example.com/image.jpg",
        "prompt": ""
    }
)

result = response.json()
if result["validation"]["is_valid"]:
    print(f"Image is valid! Quality score: {result['validation']['quality_score']}")
else:
    print(f"Issues found: {result['validation']['issues']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/validate-image-quality/" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "prompt": ""
  }'
```

## Troubleshooting

### Issue: "Gemini client not initialized"

**Solution:**
- Check that `GEMINI_API_KEY` is set in your `.env` file
- Verify the API key is valid at [aistudio.google.com](https://aistudio.google.com)
- Ensure the API key has proper permissions

### Issue: "Validation service error"

**Solution:**
- Check your internet connection
- Verify the Gemini API is enabled in Google Cloud Console
- Check API quota limits in Google Cloud Console

### Issue: "JSON parse error"

**Solution:**
- This is usually a temporary issue with the Gemini API response
- The system will default to allowing the image if validation fails
- Check logs for more details

## Alternative: Google Cloud Vision API

If you need more advanced features (like explicit content detection, text detection, etc.), you can use Google Cloud Vision API instead:

1. **Set up Google Cloud Vision API:**
   ```bash
   pip install google-cloud-vision
   ```

2. **Get credentials:**
   - Create a service account in Google Cloud Console
   - Download the JSON key file
   - Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

3. **Update the code:**
   - Replace the `validate_image_quality` function with Vision API calls
   - Use `google.cloud.vision` library

For more details, see: [Google Cloud Vision API Documentation](https://cloud.google.com/vision/docs)

## Cost Considerations

- Google AI Studio (Gemini API) offers free tier with generous limits
- Quality validation uses the text model (`gemini-1.5-pro`) which is cost-effective
- Check current pricing at [Google AI Studio Pricing](https://aistudio.google.com/pricing)

## Security Notes

- **Never commit API keys to version control**
- Use environment variables or secure secret management
- Rotate API keys regularly
- Monitor API usage in Google Cloud Console

## Support

For issues or questions:
1. Check the logs in your backend server
2. Verify API key and permissions
3. Check Google AI Studio dashboard for API status
4. Review Google Cloud Console for quota and billing

