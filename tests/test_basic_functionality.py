"""Basic functionality tests for ConvFinQA implementation."""

import pytest
from src.utils.data_loader import DataLoader
from src.models.data_models import ConvFinQARecord


def test_data_loader_initialization():
    """Test that DataLoader can be initialized."""
    loader = DataLoader()
    assert loader.dataset_path.name == "convfinqa_dataset.json"


def test_load_dataset():
    """Test loading the dataset."""
    loader = DataLoader()
    dataset = loader.load_dataset()
    
    assert isinstance(dataset, dict)
    assert "train" in dataset
    assert "dev" in dataset


def test_load_records():
    """Test loading records from dataset."""
    loader = DataLoader()
    records = loader.load_records("train")
    
    assert len(records) > 0
    assert all(isinstance(record, ConvFinQARecord) for record in records)


def test_load_specific_record():
    """Test loading a specific record by ID."""
    loader = DataLoader()
    record = loader.load_record_by_id("Single_JKHY/2009/page_28.pdf-3")
    
    assert record is not None
    assert record.id == "Single_JKHY/2009/page_28.pdf-3"
    assert len(record.dialogue.conv_questions) > 0


def test_dataset_statistics():
    """Test dataset statistics generation."""
    loader = DataLoader()
    stats = loader.get_dataset_statistics()
    
    assert "train" in stats
    assert "dev" in stats
    assert stats["train"]["total_records"] > 0
    assert "avg_dialogue_turns" in stats["train"]


def test_prompt_builder_import():
    """Test that prompt builder can be imported and initialized."""
    from src.agents.prompt_builder import FinancialPromptBuilder
    
    builder = FinancialPromptBuilder()
    assert builder is not None
    assert hasattr(builder, 'build_conversation_prompt')


def test_agent_initialization_without_api_key():
    """Test that agent initialization fails gracefully without API key.""" 
    import os
    from unittest.mock import patch
    from src.agents.conv_finqa_agent import ConvFinQAAgent
    
    # Mock load_dotenv to prevent loading from .env file
    # and ensure no API keys are available in environment
    with patch('src.agents.conv_finqa_agent.load_dotenv'):
        with patch.dict(os.environ, {}, clear=True):  # Clear all environment variables
            # Should raise an error when no API key is provided
            with pytest.raises(ValueError, match="OpenAI API key not found"):
                ConvFinQAAgent(api_key=None)


if __name__ == "__main__":
    pytest.main([__file__])
