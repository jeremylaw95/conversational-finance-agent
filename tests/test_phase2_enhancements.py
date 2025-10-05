"""Test Phase 2 enhancements: conversation intelligence and self-correction."""

import pytest
from unittest.mock import Mock, patch
from src.agents.conv_finqa_agent import ConvFinQAAgent, ConversationManager
from src.models.data_models import Document, AgentResponse


class TestConversationManager:
    """Test the enhanced conversation manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cm = ConversationManager()
    
    def test_temporal_reference_resolution(self):
        """Test temporal reference resolution like 'what about in 2008?'"""
        history = [("What is the revenue in 2009?", "100000")]
        
        resolved = self.cm.resolve_question_references("what about in 2008?", history)
        
        assert "Context:" in resolved
        assert "Same metric as previous question" in resolved or "2008" in resolved
    
    def test_calculation_reference_resolution(self):
        """Test calculation reference resolution like 'what is the difference?'"""
        history = [
            ("What is the revenue in 2009?", "100000"),
            ("What is the revenue in 2008?", "80000")
        ]
        
        resolved = self.cm.resolve_question_references("what is the difference?", history)
        
        assert "Context:" in resolved
        assert "difference" in resolved.lower() or "calculate" in resolved.lower()
    
    def test_pronoun_reference_resolution(self):
        """Test pronoun reference resolution like 'that percentage'"""
        history = [("What is the growth rate?", "15%")]
        
        resolved = self.cm.resolve_question_references("what does that percentage mean?", history)
        
        assert "Context:" in resolved
        assert "percentage" in resolved.lower()
    
    def test_conversation_state_tracking(self):
        """Test that conversation state is properly tracked."""
        history = [
            ("What is the revenue in 2009?", "100000"),
            ("What is the revenue in 2008?", "80000")
        ]
        
        self.cm._update_conversation_state(history)
        summary = self.cm.get_conversation_summary(history)
        
        assert summary["conversation_length"] == 2
        assert summary["last_metric"] == "revenue"
        assert len(summary["recent_values"]) > 0
    
    def test_metric_extraction(self):
        """Test financial metric extraction from questions."""
        test_cases = [
            ("What is the revenue in 2009?", "revenue"),
            ("Show me the cash flow data", "cash"),
            ("What are the total assets?", "assets"),
            ("How much debt do we have?", "debt")
        ]
        
        for question, expected_metric in test_cases:
            extracted = self.cm._extract_metric_from_question(question)
            assert extracted == expected_metric


class TestEnhancedAgent:
    """Test the enhanced agent with Phase 2 features."""
    
    @patch('src.agents.conv_finqa_agent.OpenAI')
    def test_enhanced_question_processing(self, mock_openai):
        """Test that enhanced question processing uses conversation manager."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
REASONING: This is a test response with proper reasoning steps.
CALCULATION: N/A
ANSWER: 12345
"""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create agent with mocked API key
        agent = ConvFinQAAgent(api_key="test-key")
        
        # Create a test document
        document = Document(
            pre_text="Test document",
            post_text="End of document", 
            table={"2009": {"revenue": 100000}, "2008": {"revenue": 80000}}
        )
        
        # Test with conversation history
        previous_qa_pairs = [("What is the revenue in 2009?", "100000")]
        
        response = agent.process_single_question(
            question="what about in 2008?",
            document=document,
            previous_qa_pairs=previous_qa_pairs,
            turn_number=2
        )
        
        assert isinstance(response, AgentResponse)
        assert response.answer is not None
        assert len(response.reasoning_steps) > 0
        assert response.confidence > 0
    
    def test_question_type_classification(self):
        """Test question type classification for self-correction."""
        from src.agents.prompt_builder import FinancialPromptBuilder
        
        builder = FinancialPromptBuilder()
        
        test_cases = [
            ("What is the revenue?", "lookup"),
            ("what about in 2008?", "reference"), 
            ("what is the difference?", "calculation"),
            ("what percentage change is that?", "percentage")
        ]
        
        for question, expected_type in test_cases:
            question_type = builder._classify_question_type(question, [])
            assert question_type == expected_type
    
    @patch('src.agents.conv_finqa_agent.OpenAI')
    def test_response_quality_validation(self, mock_openai):
        """Test response quality validation mechanism."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = ConvFinQAAgent(api_key="test-key")
        
        # Test valid response
        valid_response = """
REASONING: Clear step-by-step reasoning here.
CALCULATION: add(100, 200)
ANSWER: 300
"""
        assert agent._validate_response_quality(valid_response, "calculation") == True
        
        # Test invalid response (missing sections)
        invalid_response = "Just some text without proper structure"
        assert agent._validate_response_quality(invalid_response, "calculation") == False
        
        # Test percentage response validation
        percentage_response = """
REASONING: Calculating percentage change.
ANSWER: 15%
"""
        assert agent._validate_response_quality(percentage_response, "percentage") == True
        
        percentage_response_no_symbol = """
REASONING: Calculating percentage change.
ANSWER: 15
"""
        assert agent._validate_response_quality(percentage_response_no_symbol, "percentage") == False
    
    @patch('src.agents.conv_finqa_agent.OpenAI')
    def test_enhanced_confidence_estimation(self, mock_openai):
        """Test enhanced confidence estimation with context awareness."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        agent = ConvFinQAAgent(api_key="test-key")
        
        # Test high confidence response
        good_response = """
REASONING: Clear, detailed reasoning with multiple steps.
CALCULATION: subtract(206588, 181001)
ANSWER: 25587
"""
        parsed_good = {
            "reasoning_steps": ["Clear reasoning step 1", "Clear reasoning step 2"],
            "calculation": "subtract(206588, 181001)",
            "answer": "25587"
        }
        
        confidence = agent._estimate_enhanced_confidence(
            good_response, parsed_good, "calculation", 2
        )
        assert confidence > 0.7  # Should be high confidence
        
        # Test low confidence response
        uncertain_response = """
REASONING: I'm not sure about this calculation.
ANSWER: maybe 123
"""
        parsed_uncertain = {
            "reasoning_steps": ["I'm not sure"],
            "calculation": "",
            "answer": "maybe 123"
        }
        
        confidence = agent._estimate_enhanced_confidence(
            uncertain_response, parsed_uncertain, "calculation", 2
        )
        assert confidence < 0.6  # Should be lower confidence


def test_phase2_integration():
    """Test that Phase 2 components integrate properly."""
    # Test that we can import all Phase 2 components
    from src.agents.conv_finqa_agent import ConvFinQAAgent, ConversationManager
    from src.agents.prompt_builder import FinancialPromptBuilder
    
    # Test basic initialization
    cm = ConversationManager()
    builder = FinancialPromptBuilder()
    
    assert hasattr(cm, 'resolve_question_references')
    assert hasattr(cm, 'get_conversation_summary')
    assert hasattr(builder, 'add_self_correction_layer')
    
    # Test that conversation manager has enhanced patterns
    assert len(cm.reference_patterns) >= 6  # Should have 6 pattern categories
    assert 'conversation_state' in cm.__dict__
    
    print("✅ Phase 2 integration test passed!")


if __name__ == "__main__":
    # Run specific Phase 2 tests
    pytest.main([__file__, "-v"])
