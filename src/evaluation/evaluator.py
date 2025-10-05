"""Comprehensive evaluation framework for ConvFinQA agent."""

import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

from src.models.data_models import AgentResponse


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for conversational financial QA."""
    
    exact_match_rate: float
    numerical_accuracy: float
    conversation_success_rate: float
    self_correction_effectiveness: float
    avg_response_time: float
    error_breakdown: Dict[str, int]
    total_questions: int = 0
    total_conversations: int = 0


class ConvFinQAEvaluator:
    """Comprehensive evaluation framework for conversational financial QA."""
    
    def __init__(self):
        self.error_categories = {
            'lookup_error': 0,
            'calculation_error': 0, 
            'reference_resolution_error': 0,
            'format_error': 0,
            'reasoning_error': 0,
            'percentage_error': 0  # Added based on Phase 1 findings
        }
        self.logger = logging.getLogger(__name__)
    
    def evaluate_full_conversation(
        self,
        predicted_responses: List[AgentResponse],
        ground_truth_answers: List[str],
        ground_truth_programs: List[str],
        conversation_questions: List[str]
    ) -> EvaluationMetrics:
        """Evaluate complete conversation performance with detailed analysis."""
        
        if len(predicted_responses) != len(ground_truth_answers):
            self.logger.warning(
                f"Response count mismatch: {len(predicted_responses)} vs {len(ground_truth_answers)}"
            )
            # Pad shorter list with placeholder responses
            min_len = min(len(predicted_responses), len(ground_truth_answers))
            predicted_responses = predicted_responses[:min_len]
            ground_truth_answers = ground_truth_answers[:min_len]
            ground_truth_programs = ground_truth_programs[:min_len]
            conversation_questions = conversation_questions[:min_len]
        
        exact_matches = []
        numerical_accuracies = []
        response_times = []
        
        # Reset error categories for this conversation
        conversation_errors = {key: 0 for key in self.error_categories.keys()}
        
        for i, (pred, truth_answer, truth_program, question) in enumerate(
            zip(predicted_responses, ground_truth_answers, ground_truth_programs, conversation_questions)
        ):
            
            self.logger.debug(f"Evaluating turn {i+1}: '{question}'")
            self.logger.debug(f"Expected: '{truth_answer}', Got: '{pred.answer}'")
            
            # Exact match evaluation
            exact_match = self._evaluate_exact_match(pred.answer, truth_answer)
            exact_matches.append(exact_match)
            
            # Numerical accuracy (more lenient than exact match)
            num_accuracy = self._evaluate_numerical_accuracy(pred.answer, truth_answer)
            numerical_accuracies.append(num_accuracy)
            
            # Response time tracking
            response_times.append(pred.processing_time)
            
            # Error categorization for failed responses
            if not exact_match:
                error_type = self._categorize_error(
                    question, pred, truth_answer, truth_program, i
                )
                conversation_errors[error_type] += 1
                self.logger.debug(f"Categorized error as: {error_type}")
        
        # Update global error counts
        for error_type, count in conversation_errors.items():
            self.error_categories[error_type] += count
        
        # Conversation-level success (all turns correct)
        conversation_success = all(exact_matches) if exact_matches else False
        
        # Self-correction effectiveness analysis
        self_correction_score = self._evaluate_self_correction(predicted_responses)
        
        # Calculate average metrics
        avg_exact_match = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
        avg_numerical_accuracy = sum(numerical_accuracies) / len(numerical_accuracies) if numerical_accuracies else 0.0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        metrics = EvaluationMetrics(
            exact_match_rate=avg_exact_match,
            numerical_accuracy=avg_numerical_accuracy,
            conversation_success_rate=float(conversation_success),
            self_correction_effectiveness=self_correction_score,
            avg_response_time=avg_response_time,
            error_breakdown=conversation_errors.copy(),
            total_questions=len(predicted_responses),
            total_conversations=1
        )
        
        self.logger.info(f"Conversation evaluation complete: {avg_exact_match:.1%} exact match rate")
        return metrics
    
    def _evaluate_exact_match(self, predicted: str, ground_truth: str) -> bool:
        """Exact string match evaluation with normalization."""
        
        # Normalize both strings
        pred_normalized = self._normalize_answer(predicted)
        truth_normalized = self._normalize_answer(ground_truth)
        
        if pred_normalized == truth_normalized:
            return True
            
        # Try numerical comparison for very close values
        pred_num = self._extract_number(predicted)
        truth_num = self._extract_number(ground_truth)
        
        if pred_num is not None and truth_num is not None:
            # Handle very small differences (e.g., 108.6 vs 108.59)
            if abs(truth_num) > 0:
                relative_error = abs(pred_num - truth_num) / abs(truth_num)
                if relative_error < 0.001:  # 0.1% tolerance for exact match
                    return True
        
        return False
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""
        
        # Convert to lowercase and strip whitespace
        normalized = answer.strip().lower()
        
        # Remove common prefixes/suffixes
        normalized = re.sub(r'^(answer:\s*|result:\s*)', '', normalized)
        normalized = re.sub(r'\s*(dollars?|usd|\$|billions?|millions?|thousands?)', '', normalized)
        
        # Normalize percentages (including negative)
        normalized = re.sub(r'(-?\d+\.?\d*)\s*%', r'\1%', normalized)
        normalized = re.sub(r'(-?\d+\.?\d*)\s*percent', r'\1%', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _evaluate_numerical_accuracy(self, predicted: str, ground_truth: str) -> float:
        """Numerical accuracy within tolerance."""
        
        pred_num = self._extract_number(predicted)
        truth_num = self._extract_number(ground_truth)
        
        if pred_num is None or truth_num is None:
            # If both are non-numeric, check string similarity
            if pred_num is None and truth_num is None:
                return 1.0 if self._evaluate_exact_match(predicted, ground_truth) else 0.0
            return 0.0
            
        # Handle zero values carefully
        if truth_num == 0:
            return 1.0 if pred_num == 0 else 0.0
            
        # Calculate relative error
        relative_error = abs(pred_num - truth_num) / abs(truth_num)
        
        # Tiered accuracy scoring
        if relative_error < 0.001:  # 0.1% tolerance
            return 1.0
        elif relative_error < 0.01:  # 1% tolerance
            return 0.9
        elif relative_error < 0.05:  # 5% tolerance
            return 0.7
        elif relative_error < 0.1:   # 10% tolerance
            return 0.5
        else:
            return 0.0
    
    def _extract_number(self, text: str) -> float:
        """Extract numerical value from text with comprehensive patterns."""
        if not text:
            return None
            
        try:
            # Handle percentage values (including negative)
            percentage_match = re.search(r'(-?\d+\.?\d*)\s*%', text)
            if percentage_match:
                return float(percentage_match.group(1))
            
            # Handle percentage in decimal form (like 0.141 for 14.1%)
            decimal_percentage = re.search(r'0\.\d{3,}', text)
            if decimal_percentage:
                val = float(decimal_percentage.group())
                # If it's a small decimal, assume it's already in percentage form
                if val < 1:
                    return val
            
            # Handle regular numbers (including negative)
            number_match = re.search(r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?', text)
            if number_match:
                # Remove commas and convert
                num_str = number_match.group().replace(',', '')
                return float(num_str)
                
            # Handle scientific notation
            sci_match = re.search(r'-?\d+\.?\d*[eE][+-]?\d+', text)
            if sci_match:
                return float(sci_match.group())
                
        except (ValueError, AttributeError) as e:
            self.logger.debug(f"Number extraction failed for '{text}': {e}")
        
        return None
    
    def _categorize_error(
        self, 
        question: str, 
        predicted: AgentResponse, 
        truth_answer: str, 
        truth_program: str,
        turn_number: int
    ) -> str:
        """Categorize the type of error for detailed analysis."""
        
        q_lower = question.lower()
        
        # Check for specific error patterns based on Phase 1 findings
        
        # Percentage calculation errors (critical finding from Phase 1)
        if any(pct in q_lower for pct in ['percentage', 'percent', '%']) or '%' in truth_answer:
            return 'percentage_error'
        
        # Reference resolution errors (questions with pronouns/references)
        if turn_number > 0 and any(ref in q_lower for ref in ['what about', 'the difference', 'that', 'this']):
            return 'reference_resolution_error'
        
        # Calculation errors (has calculation operations in ground truth)
        if any(op in truth_program for op in ['add(', 'subtract(', 'divide(', 'multiply(']):
            return 'calculation_error'
        
        # Format errors (response structure issues)
        if not predicted.answer or len(predicted.reasoning_steps) == 0:
            return 'format_error'
        
        # Simple lookup errors (first turn or basic operations)
        if turn_number == 0 or not any(op in truth_program for op in ['add(', 'subtract(', 'divide(']):
            return 'lookup_error'
        
        # Default to reasoning error
        return 'reasoning_error'
    
    def _evaluate_self_correction(self, responses: List[AgentResponse]) -> float:
        """Evaluate how effectively self-correction mechanisms were used."""
        
        if not responses:
            return 0.0
        
        correction_indicators = [
            'revised', 'corrected', 'actually', 'wait', 'mistake', 
            'let me recalculate', 'on second thought', 'correction'
        ]
        
        corrections_found = 0
        high_confidence_corrections = 0
        
        for response in responses:
            raw_text = response.raw_response.lower()
            
            # Check for correction language
            if any(indicator in raw_text for indicator in correction_indicators):
                corrections_found += 1
                
                # Higher weight for corrections that led to high confidence
                if response.confidence > 0.8:
                    high_confidence_corrections += 1
        
        # Score based on appropriate use of self-correction
        if len(responses) == 0:
            return 0.0
        
        base_score = corrections_found / len(responses)
        confidence_bonus = high_confidence_corrections / len(responses) * 0.2
        
        return min(1.0, base_score + confidence_bonus)
    
    def aggregate_metrics(self, metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
        """Aggregate metrics across multiple conversations."""
        
        if not metrics_list:
            return EvaluationMetrics(
                exact_match_rate=0.0,
                numerical_accuracy=0.0,
                conversation_success_rate=0.0,
                self_correction_effectiveness=0.0,
                avg_response_time=0.0,
                error_breakdown={},
                total_questions=0,
                total_conversations=0
            )
        
        # Aggregate error breakdowns
        aggregated_errors = {}
        for metrics in metrics_list:
            for error_type, count in metrics.error_breakdown.items():
                aggregated_errors[error_type] = aggregated_errors.get(error_type, 0) + count
        
        # Weight metrics by number of questions in each conversation
        total_questions = sum(m.total_questions for m in metrics_list)
        
        if total_questions == 0:
            weighted_exact_match = 0.0
            weighted_numerical_accuracy = 0.0
            weighted_response_time = 0.0
            weighted_self_correction = 0.0
        else:
            weighted_exact_match = sum(
                m.exact_match_rate * m.total_questions for m in metrics_list
            ) / total_questions
            
            weighted_numerical_accuracy = sum(
                m.numerical_accuracy * m.total_questions for m in metrics_list
            ) / total_questions
            
            weighted_response_time = sum(
                m.avg_response_time * m.total_questions for m in metrics_list
            ) / total_questions
            
            weighted_self_correction = sum(
                m.self_correction_effectiveness * m.total_questions for m in metrics_list
            ) / total_questions
        
        # Conversation success rate
        successful_conversations = sum(1 for m in metrics_list if m.conversation_success_rate > 0.99)
        conversation_success_rate = successful_conversations / len(metrics_list)
        
        return EvaluationMetrics(
            exact_match_rate=weighted_exact_match,
            numerical_accuracy=weighted_numerical_accuracy,
            conversation_success_rate=conversation_success_rate,
            self_correction_effectiveness=weighted_self_correction,
            avg_response_time=weighted_response_time,
            error_breakdown=aggregated_errors,
            total_questions=total_questions,
            total_conversations=len(metrics_list)
        )
