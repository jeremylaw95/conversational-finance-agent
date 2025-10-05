"""Metrics reporting and analysis for ConvFinQA evaluation."""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from src.evaluation.evaluator import EvaluationMetrics


class MetricsReporter:
    """Generate comprehensive evaluation reports and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_detailed_report(
        self,
        metrics: EvaluationMetrics,
        test_conversations: List[Dict],
        model_name: str,
        test_timestamp: Optional[str] = None,
        additional_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report with actionable insights."""
        
        if not test_timestamp:
            test_timestamp = datetime.now().isoformat()
        
        # Core performance metrics
        performance_summary = {
            "exact_match_rate": f"{metrics.exact_match_rate:.3f}",
            "numerical_accuracy": f"{metrics.numerical_accuracy:.3f}",
            "conversation_success_rate": f"{metrics.conversation_success_rate:.3f}",
            "avg_response_time": f"{metrics.avg_response_time:.3f}s",
            "self_correction_effectiveness": f"{metrics.self_correction_effectiveness:.3f}"
        }
        
        # Error analysis
        error_analysis = self._analyze_error_patterns(metrics.error_breakdown, metrics.total_questions)
        
        # Performance categorization
        performance_category = self._categorize_performance(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, error_analysis)
        
        # Strengths and limitations analysis
        strengths = self._identify_strengths(metrics)
        limitations = self._identify_limitations(metrics)
        
        report = {
            "evaluation_summary": {
                "model": model_name,
                "timestamp": test_timestamp,
                "total_conversations": metrics.total_conversations,
                "total_questions": metrics.total_questions,
                "performance_category": performance_category,
                "overall_performance": performance_summary
            },
            "detailed_metrics": {
                "accuracy_breakdown": {
                    "exact_match_rate": metrics.exact_match_rate,
                    "numerical_accuracy": metrics.numerical_accuracy,
                    "gap_analysis": metrics.numerical_accuracy - metrics.exact_match_rate
                },
                "conversation_analysis": {
                    "success_rate": metrics.conversation_success_rate,
                    "avg_turns_per_conversation": metrics.total_questions / max(metrics.total_conversations, 1),
                    "multi_turn_complexity": "high" if metrics.total_questions / max(metrics.total_conversations, 1) > 4 else "medium"
                },
                "response_time_analysis": {
                    "avg_response_time": metrics.avg_response_time,
                    "performance_rating": self._rate_response_time(metrics.avg_response_time)
                }
            },
            "error_analysis": error_analysis,
            "strengths_and_limitations": {
                "key_strengths": strengths,
                "major_limitations": limitations,
                "critical_issues": self._identify_critical_issues(metrics)
            },
            "recommendations": recommendations,
            "additional_context": additional_context or {}
        }
        
        return report
    
    def _analyze_error_patterns(self, error_breakdown: Dict[str, int], total_questions: int) -> Dict[str, Any]:
        """Analyze error patterns for actionable insights."""
        
        total_errors = sum(error_breakdown.values())
        
        if total_errors == 0:
            return {
                "summary": "No errors detected - perfect performance",
                "error_distribution": {},
                "primary_failure_mode": None,
                "error_rate": 0.0
            }
        
        # Calculate error percentages
        error_percentages = {
            error_type: (count / total_errors) * 100 
            for error_type, count in error_breakdown.items() if count > 0
        }
        
        # Identify primary failure mode
        primary_failure = max(error_breakdown.items(), key=lambda x: x[1])
        
        # Generate pattern descriptions
        patterns = []
        for error_type, count in error_breakdown.items():
            if count > 0:
                percentage = (count / total_errors) * 100
                patterns.append(f"{error_type}: {count} errors ({percentage:.1f}% of all errors)")
        
        return {
            "summary": f"{total_errors} errors across {total_questions} questions ({total_errors/total_questions*100:.1f}% error rate)",
            "error_distribution": error_percentages,
            "primary_failure_mode": {
                "type": primary_failure[0],
                "count": primary_failure[1],
                "description": self._get_error_description(primary_failure[0])
            },
            "error_patterns": patterns,
            "error_rate": total_errors / total_questions
        }
    
    def _get_error_description(self, error_type: str) -> str:
        """Get human-readable description of error types."""
        
        descriptions = {
            "lookup_error": "Failed to correctly retrieve values from financial tables",
            "calculation_error": "Arithmetic operations or multi-step calculations incorrect",
            "reference_resolution_error": "Unable to resolve pronouns or contextual references",
            "format_error": "Response structure issues or parsing failures",
            "reasoning_error": "Logical reasoning flaws in step-by-step analysis",
            "percentage_error": "Percentage calculations or formatting failures"
        }
        
        return descriptions.get(error_type, f"Unknown error type: {error_type}")
    
    def _categorize_performance(self, metrics: EvaluationMetrics) -> str:
        """Categorize overall performance level."""
        
        exact_match = metrics.exact_match_rate
        
        if exact_match >= 0.9:
            return "excellent"
        elif exact_match >= 0.8:
            return "good"
        elif exact_match >= 0.7:
            return "satisfactory"
        elif exact_match >= 0.5:
            return "needs_improvement"
        else:
            return "poor"
    
    def _rate_response_time(self, avg_time: float) -> str:
        """Rate response time performance."""
        
        if avg_time < 3.0:
            return "excellent"
        elif avg_time < 5.0:
            return "good"
        elif avg_time < 10.0:
            return "acceptable"
        elif avg_time < 15.0:
            return "slow"
        else:
            return "very_slow"
    
    def _identify_strengths(self, metrics: EvaluationMetrics) -> List[str]:
        """Identify system strengths based on metrics."""
        
        strengths = []
        
        if metrics.exact_match_rate > 0.8:
            strengths.append(f"High exact match accuracy ({metrics.exact_match_rate:.1%}) - strong overall performance")
        
        if metrics.numerical_accuracy > 0.85:
            strengths.append(f"Excellent numerical precision ({metrics.numerical_accuracy:.1%}) - handles numbers accurately")
        
        if metrics.avg_response_time < 5.0:
            strengths.append(f"Fast response times ({metrics.avg_response_time:.1f}s average) - good user experience")
        
        if metrics.conversation_success_rate > 0.6:
            strengths.append(f"Good conversation coherence ({metrics.conversation_success_rate:.1%} complete success)")
        
        if metrics.self_correction_effectiveness > 0.1:
            strengths.append(f"Shows self-correction capabilities ({metrics.self_correction_effectiveness:.1%} effectiveness)")
        
        # Check error distribution for strengths
        if metrics.error_breakdown.get('lookup_error', 0) == 0:
            strengths.append("Perfect table lookup accuracy - no basic retrieval errors")
        
        if metrics.error_breakdown.get('reference_resolution_error', 0) <= 1:
            strengths.append("Strong reference resolution - handles contextual questions well")
        
        return strengths if strengths else ["System demonstrates basic functionality"]
    
    def _identify_limitations(self, metrics: EvaluationMetrics) -> List[str]:
        """Identify system limitations."""
        
        limitations = []
        
        if metrics.exact_match_rate < 0.6:
            limitations.append(f"Below target exact match accuracy ({metrics.exact_match_rate:.1%}) - needs improvement")
        
        if metrics.conversation_success_rate < 0.4:
            limitations.append(f"Struggles with multi-turn conversation coherence ({metrics.conversation_success_rate:.1%} success)")
        
        if metrics.avg_response_time > 10.0:
            limitations.append(f"Slow response times ({metrics.avg_response_time:.1f}s) may impact user experience")
        
        # Error-specific limitations
        if metrics.error_breakdown.get('percentage_error', 0) > 0:
            limitations.append("Percentage calculation issues detected - critical for financial analysis")
        
        if metrics.error_breakdown.get('calculation_error', 0) > 2:
            limitations.append("Multiple calculation errors - arithmetic accuracy needs improvement")
        
        if metrics.error_breakdown.get('reference_resolution_error', 0) > 2:
            limitations.append("Difficulty resolving conversational references - context handling needs work")
        
        return limitations
    
    def _identify_critical_issues(self, metrics: EvaluationMetrics) -> List[str]:
        """Identify critical issues requiring immediate attention."""
        
        critical_issues = []
        
        # High-priority issues based on error patterns
        if metrics.error_breakdown.get('percentage_error', 0) > 0:
            critical_issues.append("Percentage calculation failures - core financial analysis capability")
        
        if metrics.exact_match_rate < 0.5:
            critical_issues.append("Very low accuracy rate - fundamental system reliability issue")
        
        if metrics.error_breakdown.get('format_error', 0) > 1:
            critical_issues.append("Multiple response parsing failures - output structure reliability")
        
        if metrics.avg_response_time > 20.0:
            critical_issues.append("Extremely slow response times - system performance issue")
        
        return critical_issues
    
    def _generate_recommendations(self, metrics: EvaluationMetrics, error_analysis: Dict) -> List[str]:
        """Generate concrete improvement recommendations."""
        
        recommendations = []
        
        # Error-specific recommendations
        if metrics.error_breakdown.get('percentage_error', 0) > 0:
            recommendations.append("CRITICAL: Fix percentage calculation prompts and parsing logic")
        
        if metrics.error_breakdown.get('reference_resolution_error', 0) > 1:
            recommendations.append("Enhance reference resolution in conversation prompts")
        
        if metrics.error_breakdown.get('calculation_error', 0) > 1:
            recommendations.append("Add more calculation examples to few-shot prompts")
        
        if metrics.conversation_success_rate < 0.5:
            recommendations.append("Improve conversation context compression and management")
        
        if metrics.self_correction_effectiveness < 0.05:
            recommendations.append("Strengthen self-correction prompting techniques")
        
        # Performance recommendations
        if metrics.avg_response_time > 8.0:
            recommendations.append("Optimize prompt length to improve response times")
        
        # Accuracy recommendations
        gap = metrics.numerical_accuracy - metrics.exact_match_rate
        if gap > 0.1:
            recommendations.append("Focus on exact formatting - numerical accuracy is higher than exact match")
        
        return recommendations if recommendations else ["Continue current approach - performance is satisfactory"]
    
    def save_report(self, report: Dict[str, Any], output_path: str) -> None:
        """Save evaluation report to file."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Evaluation report saved to {output_path}")
    
    def print_summary(self, metrics: EvaluationMetrics) -> None:
        """Print a concise summary of evaluation results."""
        
        print("\n" + "="*60)
        print("CONVFINQA EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Overall Performance:")
        print(f"   • Exact Match Rate: {metrics.exact_match_rate:.1%}")
        print(f"   • Numerical Accuracy: {metrics.numerical_accuracy:.1%}")
        print(f"   • Conversation Success: {metrics.conversation_success_rate:.1%}")
        
        print(f"\nResponse Performance:")
        print(f"   • Average Response Time: {metrics.avg_response_time:.2f}s")
        print(f"   • Self-Correction Effectiveness: {metrics.self_correction_effectiveness:.1%}")
        
        print(f"\nTest Coverage:")
        print(f"   • Total Questions: {metrics.total_questions}")
        print(f"   • Total Conversations: {metrics.total_conversations}")
        
        if sum(metrics.error_breakdown.values()) > 0:
            print(f"\nError Breakdown:")
            for error_type, count in metrics.error_breakdown.items():
                if count > 0:
                    print(f"   • {error_type}: {count}")
        else:
            print(f"\nNo errors detected!")
        
        print("="*60)
