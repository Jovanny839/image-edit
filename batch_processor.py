"""
Batch Processing Worker
Handles processing of book generation jobs with format-specific parallelization
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from queue_manager import QueueManager, JobStatus, StageStatus, StageName
from story_lib import generate_story
import requests
import base64
from io import BytesIO
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes book generation jobs in batches"""
    
    def __init__(
        self,
        queue_manager: QueueManager,
        gemini_client,
        openai_api_key: str,
        supabase_client,
        gemini_text_model: str = "gemini-2.5-flash"
    ):
        self.queue_manager = queue_manager
        self.gemini_client = gemini_client
        self.openai_api_key = openai_api_key
        self.supabase = supabase_client
        self.storage_bucket = "images"
        self.gemini_text_model = gemini_text_model
    
    async def process_job(self, job_id: int) -> bool:
        """
        Process a single job
        
        Args:
            job_id: Job ID to process
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Claim the job
            if not self.queue_manager.claim_job(job_id):
                logger.warning(f"Could not claim job {job_id}")
                return False
            
            # Get job details
            job_status = self.queue_manager.get_job_status(job_id)
            if not job_status:
                logger.error(f"Job {job_id} not found")
                return False
            
            job = job_status["job"]
            job_type = job["job_type"]
            job_data = job["job_data"]
            
            logger.info(f"Processing job {job_id} of type {job_type}")
            
            # Process based on job type
            if job_type == "interactive_search":
                success = await self._process_interactive_search(job_id, job_data)
            elif job_type == "story_adventure":
                success = await self._process_story_adventure(job_id, job_data)
            else:
                logger.error(f"Unknown job type: {job_type}")
                self.queue_manager.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error_message=f"Unknown job type: {job_type}"
                )
                return False
            
            if success:
                self.queue_manager.update_job_status(job_id, JobStatus.COMPLETED)
                logger.info(f"Job {job_id} completed successfully")
            else:
                # Check if we should retry
                if job["retry_count"] < job["max_retries"]:
                    self.queue_manager.increment_retry_count(job_id)
                    logger.info(f"Job {job_id} will be retried (attempt {job['retry_count'] + 1}/{job['max_retries']})")
                else:
                    self.queue_manager.update_job_status(
                        job_id,
                        JobStatus.FAILED,
                        error_message="Job failed after all retries"
                    )
                    logger.error(f"Job {job_id} failed after all retries")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            self.queue_manager.update_job_status(
                job_id,
                JobStatus.FAILED,
                error_message=str(e)
            )
            return False
    
    async def _process_interactive_search(self, job_id: int, job_data: Dict[str, Any]) -> bool:
        """
        Process Interactive Search format (2 scenes simultaneously)
        
        Stages:
        1. character_extraction (M1)
        2. enhancement (M1)
        3. scene_creation (2 scenes in parallel)
        4. consistency_validation (2 scenes)
        5. pdf_creation (M3)
        """
        try:
            # Stage 1: Character Extraction
            stage_char_ext = self.queue_manager.create_stage(job_id, StageName.CHARACTER_EXTRACTION.value)
            self.queue_manager.update_stage_status(stage_char_ext["id"], StageStatus.PROCESSING)
            
            character_data = await self._extract_character(job_data)
            if not character_data:
                self.queue_manager.update_stage_status(
                    stage_char_ext["id"],
                    StageStatus.FAILED,
                    error_message="Character extraction failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_char_ext["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data=character_data
            )
            
            # Stage 2: Enhancement
            stage_enhance = self.queue_manager.create_stage(job_id, StageName.ENHANCEMENT.value)
            self.queue_manager.update_stage_status(stage_enhance["id"], StageStatus.PROCESSING)
            
            enhanced_images = await self._enhance_character(character_data, job_data)
            if not enhanced_images:
                self.queue_manager.update_stage_status(
                    stage_enhance["id"],
                    StageStatus.FAILED,
                    error_message="Character enhancement failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_enhance["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data={"enhanced_images": enhanced_images}
            )
            
            # Stage 3: Scene Creation (2 scenes in parallel)
            scene_stages = []
            for i in range(2):
                stage = self.queue_manager.create_stage(
                    job_id,
                    StageName.SCENE_CREATION.value,
                    scene_index=i
                )
                scene_stages.append(stage)
            
            # Process 2 scenes simultaneously
            scene_tasks = []
            for i, stage in enumerate(scene_stages):
                self.queue_manager.update_stage_status(stage["id"], StageStatus.PROCESSING, progress_percentage=0)
                task = self._create_scene(job_id, stage["id"], i, character_data, enhanced_images, job_data)
                scene_tasks.append(task)
            
            scene_results = await asyncio.gather(*scene_tasks, return_exceptions=True)
            
            scene_urls = []
            for i, (stage, result) in enumerate(zip(scene_stages, scene_results)):
                if isinstance(result, Exception):
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.FAILED,
                        error_message=str(result)
                    )
                    return False
                else:
                    scene_urls.append(result)
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.COMPLETED,
                        progress_percentage=100,
                        result_data={"scene_url": result}
                    )
            
            # Stage 4: Consistency Validation (2 scenes)
            validation_stages = []
            for i in range(2):
                stage = self.queue_manager.create_stage(
                    job_id,
                    StageName.CONSISTENCY_VALIDATION.value,
                    scene_index=i
                )
                validation_stages.append(stage)
            
            validation_tasks = []
            for i, stage in enumerate(validation_stages):
                self.queue_manager.update_stage_status(stage["id"], StageStatus.PROCESSING, progress_percentage=0)
                task = self._validate_consistency(job_id, stage["id"], i, scene_urls[i], enhanced_images[0] if enhanced_images else None)
                validation_tasks.append(task)
            
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            for stage, result in zip(validation_stages, validation_results):
                if isinstance(result, Exception):
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.FAILED,
                        error_message=str(result)
                    )
                else:
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.COMPLETED,
                        progress_percentage=100,
                        result_data=result
                    )
            
            # Stage 5: PDF Creation (M3)
            stage_pdf = self.queue_manager.create_stage(job_id, StageName.PDF_CREATION.value)
            self.queue_manager.update_stage_status(stage_pdf["id"], StageStatus.PROCESSING)
            
            pdf_url = await self._create_pdf(job_id, job_data, scene_urls)
            if not pdf_url:
                self.queue_manager.update_stage_status(
                    stage_pdf["id"],
                    StageStatus.FAILED,
                    error_message="PDF creation failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_pdf["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data={"pdf_url": pdf_url}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing interactive search job {job_id}: {e}")
            return False
    
    async def _process_story_adventure(self, job_id: int, job_data: Dict[str, Any]) -> bool:
        """
        Process Story Adventure format (5 scenes in parallel)
        
        Stages:
        1. character_extraction (M1)
        2. enhancement (M1)
        3. story_generation (Story Adventure only)
        4. scene_creation (5 scenes in parallel)
        5. consistency_validation (5 scenes)
        6. audio_generation (Story Adventure - M3)
        7. pdf_creation (M3)
        """
        try:
            # Stage 1: Character Extraction
            stage_char_ext = self.queue_manager.create_stage(job_id, StageName.CHARACTER_EXTRACTION.value)
            self.queue_manager.update_stage_status(stage_char_ext["id"], StageStatus.PROCESSING)
            
            character_data = await self._extract_character(job_data)
            if not character_data:
                self.queue_manager.update_stage_status(
                    stage_char_ext["id"],
                    StageStatus.FAILED,
                    error_message="Character extraction failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_char_ext["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data=character_data
            )
            
            # Stage 2: Enhancement
            stage_enhance = self.queue_manager.create_stage(job_id, StageName.ENHANCEMENT.value)
            self.queue_manager.update_stage_status(stage_enhance["id"], StageStatus.PROCESSING)
            
            enhanced_images = await self._enhance_character(character_data, job_data)
            if not enhanced_images:
                self.queue_manager.update_stage_status(
                    stage_enhance["id"],
                    StageStatus.FAILED,
                    error_message="Character enhancement failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_enhance["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data={"enhanced_images": enhanced_images}
            )
            
            # Stage 3: Story Generation
            stage_story = self.queue_manager.create_stage(job_id, StageName.STORY_GENERATION.value)
            self.queue_manager.update_stage_status(stage_story["id"], StageStatus.PROCESSING)
            
            story_result = generate_story(
                character_name=job_data.get("character_name"),
                character_type=job_data.get("character_type"),
                special_ability=job_data.get("special_ability"),
                age_group=job_data.get("age_group"),
                story_world=job_data.get("story_world"),
                adventure_type=job_data.get("adventure_type"),
                occasion_theme=job_data.get("occasion_theme"),
                use_api=True,
                api_key=self.openai_api_key
            )
            
            if not story_result:
                self.queue_manager.update_stage_status(
                    stage_story["id"],
                    StageStatus.FAILED,
                    error_message="Story generation failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_story["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data=story_result
            )
            
            # Stage 4: Scene Creation (5 scenes in parallel)
            scene_stages = []
            for i in range(5):
                stage = self.queue_manager.create_stage(
                    job_id,
                    StageName.SCENE_CREATION.value,
                    scene_index=i
                )
                scene_stages.append(stage)
            
            # Process 5 scenes simultaneously
            scene_tasks = []
            for i, stage in enumerate(scene_stages):
                self.queue_manager.update_stage_status(stage["id"], StageStatus.PROCESSING, progress_percentage=0)
                page_text = story_result["pages"][i] if i < len(story_result["pages"]) else ""
                task = self._create_story_scene(
                    job_id,
                    stage["id"],
                    i,
                    page_text,
                    character_data,
                    enhanced_images,
                    job_data
                )
                scene_tasks.append(task)
            
            scene_results = await asyncio.gather(*scene_tasks, return_exceptions=True)
            
            scene_urls = []
            for i, (stage, result) in enumerate(zip(scene_stages, scene_results)):
                if isinstance(result, Exception):
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.FAILED,
                        error_message=str(result)
                    )
                    return False
                else:
                    scene_urls.append(result)
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.COMPLETED,
                        progress_percentage=100,
                        result_data={"scene_url": result}
                    )
            
            # Stage 5: Consistency Validation (5 scenes)
            validation_stages = []
            for i in range(5):
                stage = self.queue_manager.create_stage(
                    job_id,
                    StageName.CONSISTENCY_VALIDATION.value,
                    scene_index=i
                )
                validation_stages.append(stage)
            
            validation_tasks = []
            for i, stage in enumerate(validation_stages):
                self.queue_manager.update_stage_status(stage["id"], StageStatus.PROCESSING, progress_percentage=0)
                task = self._validate_consistency(job_id, stage["id"], i, scene_urls[i], enhanced_images[0] if enhanced_images else None)
                validation_tasks.append(task)
            
            validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            for stage, result in zip(validation_stages, validation_results):
                if isinstance(result, Exception):
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.FAILED,
                        error_message=str(result)
                    )
                else:
                    self.queue_manager.update_stage_status(
                        stage["id"],
                        StageStatus.COMPLETED,
                        progress_percentage=100,
                        result_data=result
                    )
            
            # Stage 6: Audio Generation (M3)
            stage_audio = self.queue_manager.create_stage(job_id, StageName.AUDIO_GENERATION.value)
            self.queue_manager.update_stage_status(stage_audio["id"], StageStatus.PROCESSING)
            
            audio_url = await self._generate_audio(job_id, story_result)
            if not audio_url:
                self.queue_manager.update_stage_status(
                    stage_audio["id"],
                    StageStatus.SKIPPED,
                    error_message="Audio generation not implemented (M3)"
                )
            else:
                self.queue_manager.update_stage_status(
                    stage_audio["id"],
                    StageStatus.COMPLETED,
                    progress_percentage=100,
                    result_data={"audio_url": audio_url}
                )
            
            # Stage 7: PDF Creation (M3)
            stage_pdf = self.queue_manager.create_stage(job_id, StageName.PDF_CREATION.value)
            self.queue_manager.update_stage_status(stage_pdf["id"], StageStatus.PROCESSING)
            
            pdf_url = await self._create_pdf(job_id, job_data, scene_urls, story_result)
            if not pdf_url:
                self.queue_manager.update_stage_status(
                    stage_pdf["id"],
                    StageStatus.FAILED,
                    error_message="PDF creation failed"
                )
                return False
            
            self.queue_manager.update_stage_status(
                stage_pdf["id"],
                StageStatus.COMPLETED,
                progress_percentage=100,
                result_data={"pdf_url": pdf_url}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing story adventure job {job_id}: {e}")
            return False
    
    # Helper methods for processing stages
    async def _extract_character(self, job_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract character from input (M1)"""
        # Placeholder - implement character extraction logic
        return {
            "character_name": job_data.get("character_name"),
            "character_type": job_data.get("character_type"),
            "original_image_url": job_data.get("character_image_url")
        }
    
    async def _enhance_character(self, character_data: Dict[str, Any], job_data: Dict[str, Any]) -> List[str]:
        """Enhance character images (M1)"""
        # Placeholder - implement enhancement logic
        # For now, return the original image URL
        if character_data.get("original_image_url"):
            return [character_data["original_image_url"]]
        return []
    
    
    async def _create_scene(
        self,
        job_id: int,
        stage_id: int,
        scene_index: int,
        character_data: Dict[str, Any],
        enhanced_images: List[str],
        job_data: Dict[str, Any]
    ) -> str:
        """Create a scene image for Interactive Search"""
        try:
            from image_utils import generate_story_scene_image
            
            # Create a simple scene description for interactive search
            scene_description = f"Scene {scene_index + 1} featuring {job_data.get('character_name', 'the character')} in {job_data.get('story_world', 'a magical world')}"
            
            reference_image_url = enhanced_images[0] if enhanced_images else character_data.get("original_image_url")
            
            scene_url = generate_story_scene_image(
                story_page_text=scene_description,
                page_number=scene_index + 1,
                character_name=job_data.get("character_name", ""),
                character_type=job_data.get("character_type", ""),
                story_world=job_data.get("story_world", ""),
                reference_image_url=reference_image_url,
                gemini_client=self.gemini_client,
                supabase_client=self.supabase,
                storage_bucket=self.storage_bucket
            )
            
            # Update progress
            self.queue_manager.update_stage_status(stage_id, StageStatus.PROCESSING, progress_percentage=100)
            
            return scene_url
            
        except Exception as e:
            logger.error(f"Error creating scene {scene_index} for job {job_id}: {e}")
            raise
    
    async def _create_story_scene(
        self,
        job_id: int,
        stage_id: int,
        scene_index: int,
        page_text: str,
        character_data: Dict[str, Any],
        enhanced_images: List[str],
        job_data: Dict[str, Any]
    ) -> str:
        """Create a story scene image"""
        try:
            from image_utils import generate_story_scene_image
            
            reference_image_url = enhanced_images[0] if enhanced_images else character_data.get("original_image_url")
            
            scene_url = generate_story_scene_image(
                story_page_text=page_text,
                page_number=scene_index + 1,
                character_name=job_data.get("character_name", ""),
                character_type=job_data.get("character_type", ""),
                story_world=job_data.get("story_world", ""),
                reference_image_url=reference_image_url,
                gemini_client=self.gemini_client,
                supabase_client=self.supabase,
                storage_bucket=self.storage_bucket
            )
            
            # Update progress
            self.queue_manager.update_stage_status(stage_id, StageStatus.PROCESSING, progress_percentage=100)
            
            return scene_url
            
        except Exception as e:
            logger.error(f"Error creating story scene {scene_index} for job {job_id}: {e}")
            raise
    
    async def _validate_consistency(
        self,
        job_id: int,
        stage_id: int,
        scene_index: int,
        scene_url: str,
        reference_image_url: Optional[str]
    ) -> Dict[str, Any]:
        """Validate character consistency"""
        try:
            if not reference_image_url:
                # Skip validation if no reference image
                return {
                    "is_consistent": True,
                    "similarity_score": 0.5,
                    "validation_available": False,
                    "message": "No reference image provided"
                }
            
            from image_utils import download_image_from_url
            from validation_utils import validate_character_consistency
            
            # Download images
            scene_image_data = download_image_from_url(scene_url)
            reference_image_data = download_image_from_url(reference_image_url)
            
            # Validate consistency
            validation_result = validate_character_consistency(
                scene_image_data=scene_image_data,
                reference_image_data=reference_image_data,
                page_number=scene_index + 1,
                gemini_client=self.gemini_client,
                gemini_text_model=self.gemini_text_model,
                timeout_seconds=15
            )
            
            # Update progress
            self.queue_manager.update_stage_status(stage_id, StageStatus.PROCESSING, progress_percentage=100)
            
            return {
                "is_consistent": validation_result.is_consistent,
                "similarity_score": validation_result.similarity_score,
                "validation_time_seconds": validation_result.validation_time_seconds,
                "flagged": validation_result.flagged,
                "details": validation_result.details
            }
            
        except Exception as e:
            logger.error(f"Error validating consistency for scene {scene_index} in job {job_id}: {e}")
            # Return a default result on error
            return {
                "is_consistent": True,
                "similarity_score": 0.5,
                "validation_available": False,
                "error": str(e)
            }
    
    async def _generate_audio(self, job_id: int, story_result: Dict[str, Any]) -> Optional[str]:
        """Generate audio for story (M3)"""
        # Placeholder - implement audio generation logic
        return None
    
    async def _create_pdf(
        self,
        job_id: int,
        job_data: Dict[str, Any],
        scene_urls: List[str],
        story_result: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create PDF (M3)"""
        # Placeholder - implement PDF creation logic
        return None

