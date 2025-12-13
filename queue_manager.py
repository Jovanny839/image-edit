"""
Supabase-based Queue Manager
Replaces Bull Queue with Supabase as the backend storage
"""

import logging
import time
import ssl
from typing import Dict, List, Optional, Any
from datetime import datetime
from supabase import Client
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageName(str, Enum):
    CHARACTER_EXTRACTION = "character_extraction"
    ENHANCEMENT = "enhancement"
    STORY_GENERATION = "story_generation"
    SCENE_CREATION = "scene_creation"
    CONSISTENCY_VALIDATION = "consistency_validation"
    AUDIO_GENERATION = "audio_generation"
    PDF_CREATION = "pdf_creation"


class QueueManager:
    """Manages job queue using Supabase as the backend"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.max_retries = 3
        self.retry_delay = 1  # Initial delay in seconds
    
    def _is_ssl_error(self, error: Exception) -> bool:
        """Check if error is an SSL-related error"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        return (
            'ssl' in error_str or 
            'ssl' in error_type.lower() or
            'unexpected_eof' in error_str or
            'eof' in error_str or
            'connection' in error_str
        )
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff for SSL errors"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if self._is_ssl_error(e):
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"SSL error on attempt {attempt + 1}/{self.max_retries}: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"SSL error after {self.max_retries} attempts: {e}")
                else:
                    # Not an SSL error, don't retry
                    raise
        # If we get here, all retries failed
        raise last_error
    
    def create_job(
        self,
        job_type: str,
        job_data: Dict[str, Any],
        user_id: Optional[str] = None,
        child_profile_id: Optional[int] = None,
        priority: int = 5,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Create a new job in the queue
        
        Args:
            job_type: 'interactive_search' or 'story_adventure'
            job_data: Dictionary containing job parameters
            user_id: User ID (optional)
            child_profile_id: Child profile ID (optional)
            priority: Job priority (1-10, 1 is highest)
            max_retries: Maximum number of retries
        
        Returns:
            Created job record
        """
        try:
            job_record = {
                "job_type": job_type,
                "status": JobStatus.PENDING.value,
                "priority": priority,
                "max_retries": max_retries,
                "retry_count": 0,
                "job_data": job_data,
            }
            
            if user_id:
                job_record["user_id"] = user_id
            if child_profile_id:
                job_record["child_profile_id"] = child_profile_id
            
            def _execute_insert():
                return self.supabase.table("book_generation_jobs").insert(job_record).execute()
            
            result = self._retry_with_backoff(_execute_insert)
            
            if result.data and len(result.data) > 0:
                job = result.data[0]
                logger.info(f"Created job {job['id']} of type {job_type} with priority {priority}")
                return job
            else:
                raise Exception("Failed to create job: No data returned")
                
        except ssl.SSLError as ssl_error:
            logger.error(f"SSL error creating job: {ssl_error}")
            raise
        except Exception as e:
            if self._is_ssl_error(e):
                logger.error(f"SSL-related error creating job: {e}")
            else:
                logger.error(f"Error creating job: {e}")
            raise
    
    def get_next_job(self, job_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the next pending job from the queue (highest priority first)
        
        Args:
            job_type: Optional filter by job type
        
        Returns:
            Job record or None if no jobs available
        """
        try:
            def _execute_query():
                query = self.supabase.table("book_generation_jobs").select("*")
                
                if job_type:
                    query = query.eq("job_type", job_type)
                
                query = query.eq("status", JobStatus.PENDING.value)
                query = query.order("priority", desc=False)  # Lower priority number = higher priority
                query = query.order("created_at", desc=False)  # FIFO for same priority
                query = query.limit(1)
                
                return query.execute()
            
            result = self._retry_with_backoff(_execute_query)
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
            
        except ssl.SSLError as ssl_error:
            logger.error(f"SSL error getting next job: {ssl_error}")
            logger.error(f"   Error type: {type(ssl_error).__name__}")
            return None
        except Exception as e:
            if self._is_ssl_error(e):
                logger.error(f"SSL-related error getting next job: {e}")
            else:
                logger.error(f"Error getting next job: {e}")
            return None
    
    def claim_job(self, job_id: int) -> bool:
        """
        Claim a job for processing (atomically update status to processing)
        
        Args:
            job_id: Job ID to claim
        
        Returns:
            True if successfully claimed, False otherwise
        """
        try:
            def _execute_update():
                return self.supabase.table("book_generation_jobs").update({
                    "status": JobStatus.PROCESSING.value,
                    "started_at": datetime.utcnow().isoformat()
                }).eq("id", job_id).eq("status", JobStatus.PENDING.value).execute()
            
            result = self._retry_with_backoff(_execute_update)
            
            if result.data and len(result.data) > 0:
                logger.info(f"Claimed job {job_id} for processing")
                return True
            return False
            
        except ssl.SSLError as ssl_error:
            logger.error(f"SSL error claiming job {job_id}: {ssl_error}")
            return False
        except Exception as e:
            if self._is_ssl_error(e):
                logger.error(f"SSL-related error claiming job {job_id}: {e}")
            else:
                logger.error(f"Error claiming job {job_id}: {e}")
            return False
    
    def update_job_status(
        self,
        job_id: int,
        status: JobStatus,
        error_message: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update job status
        
        Args:
            job_id: Job ID
            status: New status
            error_message: Optional error message
            result_data: Optional result data
        
        Returns:
            True if successful
        """
        try:
            update_data = {"status": status.value}
            
            if error_message:
                update_data["error_message"] = error_message
            
            if result_data:
                update_data["result_data"] = result_data
            
            if status == JobStatus.COMPLETED:
                update_data["completed_at"] = datetime.utcnow().isoformat()
            
            def _execute_update():
                return self.supabase.table("book_generation_jobs").update(update_data).eq("id", job_id).execute()
            
            result = self._retry_with_backoff(_execute_update)
            
            if result.data and len(result.data) > 0:
                logger.info(f"Updated job {job_id} status to {status.value}")
                return True
            return False
            
        except ssl.SSLError as ssl_error:
            logger.error(f"SSL error updating job {job_id} status: {ssl_error}")
            return False
        except Exception as e:
            if self._is_ssl_error(e):
                logger.error(f"SSL-related error updating job {job_id} status: {e}")
            else:
                logger.error(f"Error updating job {job_id} status: {e}")
            return False
    
    def increment_retry_count(self, job_id: int) -> bool:
        """Increment retry count for a job"""
        try:
            def _get_job():
                return self.supabase.table("book_generation_jobs").select("retry_count, max_retries").eq("id", job_id).execute()
            
            job = self._retry_with_backoff(_get_job)
            
            if not job.data or len(job.data) == 0:
                return False
            
            current_retry = job.data[0]["retry_count"]
            max_retries = job.data[0]["max_retries"]
            
            new_retry_count = current_retry + 1
            
            # If exceeded max retries, mark as failed
            if new_retry_count >= max_retries:
                return self.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message=f"Job failed after {max_retries} retries"
                )
            
            def _execute_update():
                return self.supabase.table("book_generation_jobs").update({
                    "retry_count": new_retry_count,
                    "status": JobStatus.PENDING.value  # Reset to pending for retry
                }).eq("id", job_id).execute()
            
            result = self._retry_with_backoff(_execute_update)
            
            if result.data and len(result.data) > 0:
                logger.info(f"Incremented retry count for job {job_id} to {new_retry_count}")
                return True
            return False
            
        except ssl.SSLError as ssl_error:
            logger.error(f"SSL error incrementing retry count for job {job_id}: {ssl_error}")
            return False
        except Exception as e:
            if self._is_ssl_error(e):
                logger.error(f"SSL-related error incrementing retry count for job {job_id}: {e}")
            else:
                logger.error(f"Error incrementing retry count for job {job_id}: {e}")
            return False
    
    def create_stage(
        self,
        job_id: int,
        stage_name: str,
        scene_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a job stage
        
        Args:
            job_id: Job ID
            stage_name: Stage name
            scene_index: Optional scene index for scene-specific stages
        
        Returns:
            Created stage record
        """
        try:
            stage_record = {
                "job_id": job_id,
                "stage_name": stage_name,
                "status": StageStatus.PENDING.value,
                "progress_percentage": 0
            }
            
            if scene_index is not None:
                stage_record["scene_index"] = scene_index
            
            def _execute_insert():
                return self.supabase.table("job_stages").insert(stage_record).execute()
            
            result = self._retry_with_backoff(_execute_insert)
            
            if result.data and len(result.data) > 0:
                return result.data[0]
            else:
                raise Exception("Failed to create stage: No data returned")
                
        except ssl.SSLError as ssl_error:
            logger.error(f"SSL error creating stage: {ssl_error}")
            raise
        except Exception as e:
            if self._is_ssl_error(e):
                logger.error(f"SSL-related error creating stage: {e}")
            else:
                logger.error(f"Error creating stage: {e}")
            raise
    
    def update_stage_status(
        self,
        stage_id: int,
        status: StageStatus,
        progress_percentage: Optional[int] = None,
        error_message: Optional[str] = None,
        result_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update stage status
        
        Args:
            stage_id: Stage ID
            status: New status
            progress_percentage: Optional progress percentage (0-100)
            error_message: Optional error message
            result_data: Optional result data
        
        Returns:
            True if successful
        """
        try:
            update_data = {"status": status.value}
            
            if progress_percentage is not None:
                update_data["progress_percentage"] = progress_percentage
            
            if error_message:
                update_data["error_message"] = error_message
            
            if result_data:
                update_data["result_data"] = result_data
            
            if status == StageStatus.PROCESSING and "started_at" not in update_data:
                update_data["started_at"] = datetime.utcnow().isoformat()
            
            if status in [StageStatus.COMPLETED, StageStatus.FAILED, StageStatus.SKIPPED]:
                update_data["completed_at"] = datetime.utcnow().isoformat()
            
            def _execute_update():
                return self.supabase.table("job_stages").update(update_data).eq("id", stage_id).execute()
            
            result = self._retry_with_backoff(_execute_update)
            
            if result.data and len(result.data) > 0:
                logger.info(f"Updated stage {stage_id} status to {status.value}")
                return True
            return False
            
        except ssl.SSLError as ssl_error:
            logger.error(f"SSL error updating stage {stage_id} status: {ssl_error}")
            return False
        except Exception as e:
            if self._is_ssl_error(e):
                logger.error(f"SSL-related error updating stage {stage_id} status: {e}")
            else:
                logger.error(f"Error updating stage {stage_id} status: {e}")
            return False
    
    def get_job_stages(self, job_id: int) -> List[Dict[str, Any]]:
        """Get all stages for a job"""
        try:
            def _get_stages():
                return self.supabase.table("job_stages").select("*").eq("job_id", job_id).order("created_at", desc=False).execute()
            
            result = self._retry_with_backoff(_get_stages)
            return result.data if result.data else []
        except ssl.SSLError as ssl_error:
            logger.error(f"SSL error getting job stages for {job_id}: {ssl_error}")
            return []
        except Exception as e:
            if self._is_ssl_error(e):
                logger.error(f"SSL-related error getting job stages for {job_id}: {e}")
            else:
                logger.error(f"Error getting job stages for {job_id}: {e}")
            return []
    
    def get_job_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Get job status with all stages"""
        try:
            def _get_job():
                return self.supabase.table("book_generation_jobs").select("*").eq("id", job_id).execute()
            
            job_result = self._retry_with_backoff(_get_job)
            
            if not job_result.data or len(job_result.data) == 0:
                return None
            
            job = job_result.data[0]
            
            # Get stages
            stages = self.get_job_stages(job_id)
            
            # Calculate overall progress
            if stages:
                completed_stages = sum(1 for s in stages if s["status"] == StageStatus.COMPLETED.value)
                total_stages = len(stages)
                overall_progress = int((completed_stages / total_stages) * 100) if total_stages > 0 else 0
            else:
                overall_progress = 0
            
            return {
                "job": job,
                "stages": stages,
                "overall_progress": overall_progress
            }
            
        except ssl.SSLError as ssl_error:
            logger.error(f"SSL error getting job status for {job_id}: {ssl_error}")
            return None
        except Exception as e:
            if self._is_ssl_error(e):
                logger.error(f"SSL-related error getting job status for {job_id}: {e}")
            else:
                logger.error(f"Error getting job status for {job_id}: {e}")
            return None

