"""Data loading utilities for ConvFinQA dataset."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import random

from src.models.data_models import ConvFinQARecord


class DataLoader:
    """Loads and manages ConvFinQA dataset."""
    
    def __init__(self, dataset_path: str = "data/convfinqa_dataset.json"):
        self.dataset_path = Path(dataset_path)
        self.logger = logging.getLogger(__name__)
        self._dataset_cache: Optional[Dict[str, Any]] = None
        
    def load_dataset(self) -> Dict[str, Any]:
        """Load the complete dataset from JSON file."""
        if self._dataset_cache is None:
            try:
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    self._dataset_cache = json.load(f)
                self.logger.info(f"Loaded dataset from {self.dataset_path}")
            except FileNotFoundError:
                self.logger.error(f"Dataset file not found: {self.dataset_path}")
                raise
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON: {e}")
                raise
        
        return self._dataset_cache
    
    def load_records(self, split: str = "train") -> List[ConvFinQARecord]:
        """Load records from a specific split (train/dev/test)."""
        dataset = self.load_dataset()
        
        if split not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(f"Split '{split}' not found. Available splits: {available_splits}")
        
        records = []
        for record_data in dataset[split]:
            try:
                record = ConvFinQARecord(**record_data)
                records.append(record)
            except Exception as e:
                self.logger.warning(f"Skipping invalid record {record_data.get('id', 'unknown')}: {e}")
                
        self.logger.info(f"Loaded {len(records)} records from {split} split")
        return records
    
    def load_record_by_id(self, record_id: str) -> Optional[ConvFinQARecord]:
        """Load a specific record by ID."""
        dataset = self.load_dataset()
        
        for split_name, split_data in dataset.items():
            for record_data in split_data:
                if record_data.get('id') == record_id:
                    try:
                        return ConvFinQARecord(**record_data)
                    except Exception as e:
                        self.logger.error(f"Error loading record {record_id}: {e}")
                        return None
        
        self.logger.warning(f"Record {record_id} not found")
        return None
    
    def load_test_samples(self, num_samples: int = 10, split: str = "train") -> List[ConvFinQARecord]:
        """Load a random sample of records for testing."""
        all_records = self.load_records(split)
        
        if num_samples >= len(all_records):
            return all_records
        
        sample_records = random.sample(all_records, num_samples)
        self.logger.info(f"Selected {len(sample_records)} random samples from {split}")
        return sample_records
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        dataset = self.load_dataset()
        stats = {}
        
        for split_name, split_data in dataset.items():
            split_stats = {
                "total_records": len(split_data),
                "dialogue_turns": [],
                "question_types": {
                    "type2_questions": 0,
                    "duplicate_columns": 0,
                    "non_numeric_values": 0
                }
            }
            
            for record_data in split_data:
                # Dialogue turn statistics
                if 'features' in record_data:
                    features = record_data['features']
                    split_stats["dialogue_turns"].append(features.get('num_dialogue_turns', 0))
                    
                    if features.get('has_type2_question', False):
                        split_stats["question_types"]["type2_questions"] += 1
                    if features.get('has_duplicate_columns', False):
                        split_stats["question_types"]["duplicate_columns"] += 1
                    if features.get('has_non_numeric_values', False):
                        split_stats["question_types"]["non_numeric_values"] += 1
            
            # Calculate averages
            if split_stats["dialogue_turns"]:
                split_stats["avg_dialogue_turns"] = sum(split_stats["dialogue_turns"]) / len(split_stats["dialogue_turns"])
                split_stats["max_dialogue_turns"] = max(split_stats["dialogue_turns"])
                split_stats["min_dialogue_turns"] = min(split_stats["dialogue_turns"])
            
            stats[split_name] = split_stats
        
        return stats
    
    def get_sample_conversation(self, record_id: Optional[str] = None) -> Optional[ConvFinQARecord]:
        """Get a sample conversation for demonstration."""
        if record_id:
            return self.load_record_by_id(record_id)
        else:
            # Get first record from train split
            records = self.load_records("train")
            return records[0] if records else None
    
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate dataset structure and report issues."""
        validation_results = {
            "valid_records": 0,
            "invalid_records": 0,
            "errors": []
        }
        
        dataset = self.load_dataset()
        
        for split_name, split_data in dataset.items():
            for i, record_data in enumerate(split_data):
                try:
                    record = ConvFinQARecord(**record_data)
                    validation_results["valid_records"] += 1
                    
                    # Additional validation checks
                    if len(record.dialogue.conv_questions) != len(record.dialogue.conv_answers):
                        validation_results["errors"].append(
                            f"Record {record.id}: Mismatched questions/answers count"
                        )
                    
                    if len(record.dialogue.conv_questions) != record.features.num_dialogue_turns:
                        validation_results["errors"].append(
                            f"Record {record.id}: Features dialogue turns mismatch"
                        )
                        
                except Exception as e:
                    validation_results["invalid_records"] += 1
                    validation_results["errors"].append(
                        f"Split {split_name}, record {i}: {str(e)}"
                    )
        
        return validation_results

