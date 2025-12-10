# Batch Processing System Documentation

## Overview

This document describes the batch processing system implemented for book generation jobs. The system uses Supabase as the queue backend (replacing Redis/Bull Queue) and supports format-specific parallel processing.

## Architecture

### Components

1. **Queue Manager** (`queue_manager.py`)
   - Manages job queue using Supabase as the backend
   - Handles job creation, claiming, status updates, and retry logic
   - Manages job stages for progress tracking

2. **Batch Processor** (`batch_processor.py`)
   - Processes book generation jobs
   - Implements format-specific parallelization
   - Handles all processing stages

3. **Database Schema** (`job_queue_schema.sql`)
   - `book_generation_jobs` table: Stores job metadata
   - `job_stages` table: Tracks individual stage progress

## Database Setup

Run the SQL script in your Supabase SQL Editor:

```sql
-- See backend/job_queue_schema.sql
```

This creates:
- `book_generation_jobs` table with job metadata
- `job_stages` table for stage-by-stage progress tracking
- Indexes for performance
- Row Level Security (RLS) policies

## Job Types

### 1. Interactive Search
- **Parallel Scenes**: 2 scenes simultaneously
- **Stages**:
  1. `character_extraction` (M1)
  2. `enhancement` (M1)
  3. `scene_creation` (2 scenes in parallel)
  4. `consistency_validation` (2 scenes)
  5. `pdf_creation` (M3)

### 2. Story Adventure
- **Parallel Scenes**: 5 scenes simultaneously
- **Stages**:
  1. `character_extraction` (M1)
  2. `enhancement` (M1)
  3. `story_generation` (Story Adventure only)
  4. `scene_creation` (5 scenes in parallel)
  5. `consistency_validation` (5 scenes)
  6. `audio_generation` (Story Adventure - M3)
  7. `pdf_creation` (M3)

## API Endpoints

### Create Job

```http
POST /api/books/generate
Content-Type: application/json

{
  "job_type": "interactive_search" | "story_adventure",
  "character_name": "Luna",
  "character_type": "a brave dragon",
  "special_ability": "fly through clouds",
  "age_group": "7-10",
  "story_world": "the Enchanted Forest",
  "adventure_type": "treasure hunt",
  "occasion_theme": null,
  "character_image_url": "https://...",
  "priority": 5,
  "user_id": "uuid",
  "child_profile_id": 123
}
```

**Response:**
```json
{
  "success": true,
  "job_id": 1,
  "message": "Job 1 created successfully"
}
```

### Get Job Status

```http
GET /api/books/{book_id}/status
```

**Response:**
```json
{
  "job_id": 1,
  "status": "processing",
  "overall_progress": 45,
  "stages": [
    {
      "id": 1,
      "stage_name": "character_extraction",
      "status": "completed",
      "progress_percentage": 100,
      "started_at": "2024-01-01T12:00:00Z",
      "completed_at": "2024-01-01T12:00:05Z"
    },
    {
      "id": 2,
      "stage_name": "enhancement",
      "status": "processing",
      "progress_percentage": 50,
      "started_at": "2024-01-01T12:00:05Z"
    }
  ],
  "error_message": null,
  "result_data": null
}
```

## Job Status Values

- `pending`: Job is waiting to be processed
- `processing`: Job is currently being processed
- `completed`: Job completed successfully
- `failed`: Job failed after all retries
- `cancelled`: Job was cancelled

## Stage Status Values

- `pending`: Stage is waiting to be processed
- `processing`: Stage is currently being processed
- `completed`: Stage completed successfully
- `failed`: Stage failed
- `skipped`: Stage was skipped (e.g., optional stages)

## Priority System

Jobs have a priority from 1-10:
- **1**: Highest priority (processed first)
- **10**: Lowest priority (processed last)

Within the same priority, jobs are processed FIFO (First In, First Out).

## Retry Logic

- Each job has a `max_retries` value (default: 3)
- When a job fails, the retry count is incremented
- If retry count < max_retries, job is reset to `pending` status
- If retry count >= max_retries, job is marked as `failed`

## Error Handling

- Each stage can have an `error_message` field
- Failed stages are logged with specific error messages
- Jobs track overall error state in `error_message` field
- Retry logic automatically handles transient failures

## Background Worker

The system includes a background worker that:
- Continuously polls for pending jobs
- Claims jobs atomically (prevents duplicate processing)
- Processes jobs using the batch processor
- Updates job and stage status throughout processing

The worker runs automatically when the FastAPI application starts.

## Progress Tracking

Progress is tracked at two levels:

1. **Job Level**: Overall progress percentage based on completed stages
2. **Stage Level**: Individual stage progress (0-100%)

The status API returns both job-level and stage-level progress information.

## Implementation Notes

### M1, M3 Stages

- **M1**: Implemented (character_extraction, enhancement)
- **M3**: Placeholder (audio_generation, pdf_creation)
  - These stages are created but marked as `skipped` or return placeholder results
  - Implement these when M3 features are ready

### Parallel Processing

- Scene creation and validation stages run in parallel using `asyncio.gather()`
- Interactive Search: 2 scenes in parallel
- Story Adventure: 5 scenes in parallel

### Supabase Integration

- Uses Supabase as the queue backend (no Redis required)
- Leverages Supabase's PostgreSQL database for job storage
- Row Level Security (RLS) ensures users can only access their own jobs

## Usage Example

```python
# Create a job
response = requests.post("http://localhost:8000/api/books/generate", json={
    "job_type": "story_adventure",
    "character_name": "Luna",
    "character_type": "a brave dragon",
    "special_ability": "fly through clouds",
    "age_group": "7-10",
    "story_world": "the Enchanted Forest",
    "adventure_type": "treasure hunt",
    "priority": 1
})

job_id = response.json()["job_id"]

# Check status
status = requests.get(f"http://localhost:8000/api/books/{job_id}/status")
print(status.json())
```

## Future Enhancements

1. **M3 Implementation**: Implement audio generation and PDF creation
2. **Webhooks**: Add webhook support for job completion notifications
3. **Job Cancellation**: Add endpoint to cancel pending/processing jobs
4. **Job History**: Add endpoint to list user's job history
5. **Rate Limiting**: Add rate limiting per user
6. **Job Scheduling**: Add support for scheduled jobs

