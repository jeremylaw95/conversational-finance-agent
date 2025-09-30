"""Main conversational financial QA agent."""

import time
import logging
import os
import re
import asyncio
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    # Handle case where OpenAI is not installed for testing
    OpenAI = None
    AsyncOpenAI = None

from src.models.data_models import (
    Document, Dialogue, AgentResponse, ConvFinQARecord
)
from src.agents.prompt_builder import FinancialPromptBuilder


class ConvFinQAAgent:
    """Main conversational financial QA agent with enhanced conversation intelligence."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o",  # Changed to GPT-4o for reliability (was gpt-5)
        api_key: Optional[str] = None,
        temperature: float = 0.0,  # Start with deterministic responses
        max_tokens: int = 1000
    ):
        # Load API key from environment if not provided
        load_dotenv()
        
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
        
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Please install with: pip install openai")
            
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize async client if available
        if AsyncOpenAI is not None:
            self.async_client = AsyncOpenAI(api_key=api_key)
        else:
            self.async_client = None
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_builder = FinancialPromptBuilder()
        self.conversation_manager = ConversationManager()  # Add conversation manager
        self.logger = logging.getLogger(__name__)
        
    def process_conversation(
        self, 
        record: ConvFinQARecord
    ) -> List[AgentResponse]:
        """Process complete conversation sequence from dataset record."""
        
        responses = []
        previous_qa_pairs = []  # (question, answer) pairs for context
        
        for i, question in enumerate(record.dialogue.conv_questions):
            self.logger.info(f"Processing question {i+1}: {question}")
            
            response = self.process_single_question(
                question=question,
                document=record.doc,
                previous_qa_pairs=previous_qa_pairs,
                turn_number=i + 1
            )
            
            responses.append(response)
            
            # Update conversation context with our response
            previous_qa_pairs.append((question, response.answer))
            
        return responses
    
    async def process_conversation_async(
        self, 
        record: ConvFinQARecord
    ) -> List[AgentResponse]:
        """Process complete conversation sequence asynchronously."""
        
        if self.async_client is None:
            # Fall back to sync processing if async client not available
            self.logger.warning("Async client not available, falling back to sync processing")
            return self.process_conversation(record)
        
        responses = []
        previous_qa_pairs = []  # (question, answer) pairs for context
        
        for i, question in enumerate(record.dialogue.conv_questions):
            self.logger.info(f"Processing question {i+1}: {question}")
            
            response = await self.process_single_question_async(
                question=question,
                document=record.doc,
                previous_qa_pairs=previous_qa_pairs,
                turn_number=i + 1
            )
            
            responses.append(response)
            
            # Update conversation context with our response
            previous_qa_pairs.append((question, response.answer))
            
        return responses
    
    def process_single_question(
        self,
        question: str,
        document: Document, 
        previous_qa_pairs: List[tuple[str, str]],
        turn_number: int
    ) -> AgentResponse:
        """Process a single question with enhanced context management and self-correction."""
        
        start_time = time.time()
        
        try:
            # 1. Enhanced reference resolution
            resolved_question = self.conversation_manager.resolve_question_references(
                question, previous_qa_pairs
            )
            
            self.logger.info(f"Original question: {question}")
            if resolved_question != question:
                self.logger.info(f"Resolved question: {resolved_question}")
            
            # 2. Build sophisticated prompt with resolved context
            prompt = self.prompt_builder.build_conversation_prompt(
                question=resolved_question,
                document=document,
                previous_qa_pairs=previous_qa_pairs
            )
            
            # 3. Add enhanced self-correction instructions  
            question_type = self.prompt_builder._classify_question_type(question, previous_qa_pairs)
            enhanced_prompt = self._add_enhanced_self_correction_prompt(prompt, question_type, turn_number)
            
            # 4. Get LLM response with error detection
            raw_response = self._call_llm_with_validation(enhanced_prompt, question_type)
            
            # 5. Parse and validate response
            parsed_response = self._parse_llm_response(raw_response)
            
            # 6. Enhanced confidence estimation
            confidence = self._estimate_enhanced_confidence(
                raw_response, parsed_response, question_type, turn_number
            )
            
            processing_time = time.time() - start_time
            
            # 7. Log conversation state for debugging
            if self.logger.isEnabledFor(logging.DEBUG):
                conv_summary = self.conversation_manager.get_conversation_summary(previous_qa_pairs)
                self.logger.debug(f"Conversation state: {conv_summary}")
            
            return AgentResponse(
                answer=parsed_response["answer"],
                reasoning_steps=parsed_response["reasoning_steps"],
                confidence=confidence,
                raw_response=raw_response,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return AgentResponse(
                answer="ERROR",
                reasoning_steps=[f"Processing error: {str(e)}"],
                confidence=0.0,
                raw_response="",
                processing_time=time.time() - start_time
            )
    
    async def process_single_question_async(
        self,
        question: str,
        document: Document, 
        previous_qa_pairs: List[tuple[str, str]],
        turn_number: int
    ) -> AgentResponse:
        """Process a single question asynchronously with enhanced context management and self-correction."""
        
        start_time = time.time()
        
        try:
            # 1. Enhanced reference resolution
            resolved_question = self.conversation_manager.resolve_question_references(
                question, previous_qa_pairs
            )
            
            self.logger.info(f"Original question: {question}")
            if resolved_question != question:
                self.logger.info(f"Resolved question: {resolved_question}")
            
            # 2. Build sophisticated prompt with resolved context
            prompt = self.prompt_builder.build_conversation_prompt(
                question=resolved_question,
                document=document,
                previous_qa_pairs=previous_qa_pairs
            )
            
            # 3. Add enhanced self-correction instructions  
            question_type = self.prompt_builder._classify_question_type(question, previous_qa_pairs)
            enhanced_prompt = self._add_enhanced_self_correction_prompt(prompt, question_type, turn_number)
            
            # 4. Get LLM response with error detection (async)
            raw_response = await self._call_llm_with_validation_async(enhanced_prompt, question_type)
            
            # 5. Parse and validate response
            parsed_response = self._parse_llm_response(raw_response)
            
            # 6. Enhanced confidence estimation
            confidence = self._estimate_enhanced_confidence(
                raw_response, parsed_response, question_type, turn_number
            )
            
            processing_time = time.time() - start_time
            
            # 7. Log conversation state for debugging
            if self.logger.isEnabledFor(logging.DEBUG):
                conv_summary = self.conversation_manager.get_conversation_summary(previous_qa_pairs)
                self.logger.debug(f"Conversation state: {conv_summary}")
            
            return AgentResponse(
                answer=parsed_response["answer"],
                reasoning_steps=parsed_response["reasoning_steps"],
                confidence=confidence,
                raw_response=raw_response,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return AgentResponse(
                answer="ERROR",
                reasoning_steps=[f"Processing error: {str(e)}"],
                confidence=0.0,
                raw_response="",
                processing_time=time.time() - start_time
            )
    
    def _add_self_correction_prompt(self, base_prompt: str) -> str:
        """Add self-correction instructions to prompt."""
        return base_prompt + """

SELF-VALIDATION CHECKLIST:
Before providing your final answer, verify:
1. Did I correctly identify what the question is asking?
2. If this references previous context, did I resolve it correctly?
3. Did I find the right data in the document?
4. Are my calculations correct and in the right format?
5. Does my final answer make sense given the context?

If any step seems wrong, revise your reasoning before answering.
"""
    
    def _call_llm(self, prompt: str) -> str:
        """Make API call to LLM."""
        try:
            # GPT-5 models use different parameters and have known issues with some prompts
            if self.model_name.startswith("gpt-5"):
                # First try with GPT-5
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=self.max_tokens,
                        timeout=30
                        # Note: temperature must be default (1) for GPT-5 models
                    )
                    result = response.choices[0].message.content
                    
                    # If GPT-5 returns empty response, fall back to GPT-4o
                    if not result or not result.strip():
                        self.logger.warning("GPT-5 returned empty response, falling back to GPT-4o")
                        fallback_response = self.client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            timeout=30
                        )
                        result = fallback_response.choices[0].message.content
                        self.logger.info("Successfully used GPT-4o fallback")
                    
                    return result
                    
                except Exception as gpt5_error:
                    self.logger.warning(f"GPT-5 call failed: {gpt5_error}, falling back to GPT-4o")
                    # Fall back to GPT-4o
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=30
                    )
                    return response.choices[0].message.content
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=30
                )
                return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise
    
    async def _call_llm_async(self, prompt: str) -> str:
        """Make async API call to LLM."""
        if self.async_client is None:
            # Fall back to sync call
            return self._call_llm(prompt)
            
        try:
            # GPT-5 models use different parameters and have known issues with some prompts
            if self.model_name.startswith("gpt-5"):
                # First try with GPT-5
                try:
                    response = await self.async_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=self.max_tokens,
                        timeout=30
                        # Note: temperature must be default (1) for GPT-5 models
                    )
                    result = response.choices[0].message.content
                    
                    # If GPT-5 returns empty response, fall back to GPT-4o
                    if not result or not result.strip():
                        self.logger.warning("GPT-5 returned empty response, falling back to GPT-4o")
                        fallback_response = await self.async_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            timeout=30
                        )
                        result = fallback_response.choices[0].message.content
                        self.logger.info("Successfully used GPT-4o fallback")
                    
                    return result
                    
                except Exception as gpt5_error:
                    self.logger.warning(f"GPT-5 call failed: {gpt5_error}, falling back to GPT-4o")
                    # Fall back to GPT-4o
                    response = await self.async_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=30
                    )
                    return response.choices[0].message.content
            else:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=30
                )
                return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise
    
    async def _call_llm_with_validation_async(self, prompt: str, question_type: str) -> str:
        """Enhanced async LLM call with validation and error detection."""
        
        # First attempt
        try:
            response = await self._call_llm_async(prompt)
            
            # Validate response quality
            if self._validate_response_quality(response, question_type):
                return response
            else:
                self.logger.warning(f"Response quality check failed for {question_type} question, attempting retry...")
                
                # Add additional validation prompt and retry
                enhanced_prompt = prompt + f"""

⚠️ IMPORTANT: Your previous response may have had quality issues. Please ensure:
- Clear REASONING section with step-by-step logic
- Proper CALCULATION section with correct format if needed  
- Concise ANSWER section with the final result
- Double-check all numerical values and calculations
"""
                
                retry_response = await self._call_llm_async(enhanced_prompt)
                return retry_response
                
        except Exception as e:
            self.logger.error(f"Enhanced LLM call failed: {e}")
            # Fall back to basic call
            return await self._call_llm_async(prompt)
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse structured response from LLM."""
        
        # Debug: Log the raw response for troubleshooting
        self.logger.debug(f"Raw LLM response: {response}")
        
        # Handle empty response
        if not response or not response.strip():
            self.logger.warning("Empty response from LLM")
            return {
                "reasoning_steps": ["Empty response received from LLM"],
                "calculation": "",
                "answer": ""
            }
        
        # Extract sections
        sections = {}
        current_section = None
        current_content = []
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('REASONING:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'reasoning'
                current_content = [line[10:].strip()]
            elif line.startswith('CALCULATION:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'calculation'
                current_content = [line[12:].strip()]
            elif line.startswith('ANSWER:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'answer'
                current_content = [line[7:].strip()]
            elif current_section:
                current_content.append(line)
        
        # Add final section
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        # Debug: Log parsed sections
        self.logger.debug(f"Parsed sections: {sections}")
        
        # Extract reasoning steps
        reasoning_text = sections.get('reasoning', '')
        reasoning_steps = [
            step.strip() for step in reasoning_text.split('\n') 
            if step.strip()
        ]
        
        # Clean answer
        answer = sections.get('answer', '').strip()
        
        # If no structured response found, try to extract any answer from the raw response
        if not answer and not reasoning_steps:
            self.logger.warning("No structured response found, attempting fallback parsing")
            # Look for percentage values in the raw response
            import re
            percentage_match = re.search(r'(\d+\.?\d*)\s*%', response)
            if percentage_match:
                answer = percentage_match.group(0)
                reasoning_steps = ["Extracted percentage from unstructured response"]
        
        return {
            "reasoning_steps": reasoning_steps,
            "calculation": sections.get('calculation', ''),
            "answer": answer
        }
    
    def _estimate_confidence(self, raw_response: str, parsed_response: Dict[str, Any]) -> float:
        """Estimate confidence based on response characteristics."""
        
        confidence = 1.0
        
        # Reduce confidence for uncertain language
        uncertain_phrases = ['not sure', 'might be', 'unclear', 'difficult to determine']
        for phrase in uncertain_phrases:
            if phrase in raw_response.lower():
                confidence -= 0.2
        
        # Reduce confidence for missing sections
        if not parsed_response.get('reasoning_steps'):
            confidence -= 0.3
        if not parsed_response.get('answer'):
            confidence -= 0.5
            
        # Increase confidence for clear calculations
        if 'subtract(' in raw_response or 'add(' in raw_response or 'divide(' in raw_response:
            confidence += 0.1
            
        return max(0.0, min(1.0, confidence))
    
    def _add_enhanced_self_correction_prompt(self, base_prompt: str, question_type: str, turn_number: int) -> str:
        """Add enhanced self-correction instructions based on question type and conversation context."""
        
        # Base self-validation
        base_validation = """

ENHANCED SELF-VALIDATION PROTOCOL:
Before providing your final answer, systematically verify:

1. ✓ QUESTION UNDERSTANDING: What specific information is being requested?
2. ✓ CONTEXT RESOLUTION: If this references previous context, what exactly does it refer to?
3. ✓ DATA LOCATION: Did I find the correct data in the document?
4. ✓ CALCULATION ACCURACY: Are my calculations correct and properly formatted?
5. ✓ ANSWER VALIDATION: Does my final answer make logical sense?
6. ✓ FORMAT COMPLIANCE: Is my response in the required REASONING/CALCULATION/ANSWER format?
"""
        
        # Add specific validation based on question type
        type_specific_validation = {
            'lookup': """
LOOKUP-SPECIFIC CHECKS:
- Did I identify the exact metric name correctly?
- Did I use the correct time period/year?
- Is the numerical value exactly as shown in the table?
- Did I avoid confusing similar metric names?""",
            
            'reference': """
REFERENCE-SPECIFIC CHECKS:
- What specific metric from previous questions does this refer to?
- Am I using the same calculation method as established in context?
- Did I identify the correct time period being referenced?
- Does my interpretation align with the conversation flow?""",
            
            'calculation': """
CALCULATION-SPECIFIC CHECKS:
- Are my calculation steps in logical order?
- Did I use the correct operation format: operation(arg1, arg2)?
- Do the input values match what was discussed previously?
- Is my arithmetic correct (double-check manually)?
- Did I reference intermediate results correctly (#0, #1, etc.)?""",
            
            'percentage': """
PERCENTAGE-SPECIFIC CHECKS:
- Did I identify the correct base value for percentage calculation?
- Is my formula correct: (part / total) * 100 or (difference / base_value) * 100?
- Did I format the final answer with % symbol?
- Does the percentage magnitude make sense (typically 0-100% for portions/ratios)?
- Are my intermediate calculations accurate?
- Am I calculating a percentage, not just extracting a raw number?"""
        }
        
        # Add turn-specific considerations
        turn_specific = ""
        if turn_number > 1:
            turn_specific = f"""
CONVERSATION CONTEXT CHECKS (Turn {turn_number}):
- Does my answer build logically on previous responses?
- Am I maintaining consistency with previously established values?
- If this is a follow-up question, does my interpretation make sense?
"""
        
        # Combine all validations
        full_validation = base_validation + type_specific_validation.get(question_type, "") + turn_specific
        
        return base_prompt + full_validation + """
🚨 CRITICAL: If ANY validation step fails, STOP and revise your reasoning before providing the final answer.
"""
    
    def _call_llm_with_validation(self, prompt: str, question_type: str) -> str:
        """Enhanced LLM call with validation and error detection."""
        
        # First attempt
        try:
            response = self._call_llm(prompt)
            
            # Validate response quality
            if self._validate_response_quality(response, question_type):
                return response
            else:
                self.logger.warning(f"Response quality check failed for {question_type} question, attempting retry...")
                
                # Add additional validation prompt and retry
                enhanced_prompt = prompt + f"""

⚠️ IMPORTANT: Your previous response may have had quality issues. Please ensure:
- Clear REASONING section with step-by-step logic
- Proper CALCULATION section with correct format if needed  
- Concise ANSWER section with the final result
- Double-check all numerical values and calculations
"""
                
                retry_response = self._call_llm(enhanced_prompt)
                return retry_response
                
        except Exception as e:
            self.logger.error(f"Enhanced LLM call failed: {e}")
            # Fall back to basic call
            return self._call_llm(prompt)
    
    def _validate_response_quality(self, response: str, question_type: str) -> bool:
        """Validate response quality based on question type."""
        
        if not response or len(response.strip()) < 10:
            return False
        
        # Check for required sections
        has_reasoning = 'REASONING:' in response
        has_answer = 'ANSWER:' in response
        
        if not (has_reasoning and has_answer):
            return False
        
        # Type-specific validation
        if question_type == 'calculation':
            # Should have calculation section for calculation questions
            has_calculation = 'CALCULATION:' in response
            if not has_calculation:
                return False
        
        elif question_type == 'percentage':
            # Should have % symbol in answer for percentage questions
            answer_section = response.split('ANSWER:')[-1] if 'ANSWER:' in response else ""
            if '%' not in answer_section and 'percent' not in answer_section.lower():
                return False
        
        return True
    
    def _estimate_enhanced_confidence(
        self, 
        raw_response: str, 
        parsed_response: Dict[str, Any], 
        question_type: str, 
        turn_number: int
    ) -> float:
        """Enhanced confidence estimation with context awareness."""
        
        confidence = 1.0
        
        # Base confidence checks
        uncertain_phrases = ['not sure', 'might be', 'unclear', 'difficult to determine', 'uncertain']
        for phrase in uncertain_phrases:
            if phrase in raw_response.lower():
                confidence -= 0.15
        
        # Structural completeness
        if not parsed_response.get('reasoning_steps'):
            confidence -= 0.25
        if not parsed_response.get('answer'):
            confidence -= 0.4
        
        # Type-specific confidence adjustments
        if question_type == 'calculation':
            # Higher confidence for clear calculation format
            if 'subtract(' in raw_response or 'add(' in raw_response or 'divide(' in raw_response:
                confidence += 0.1
            # Lower confidence if calculation section missing
            if not parsed_response.get('calculation'):
                confidence -= 0.2
        
        elif question_type == 'percentage':
            # Check for percentage formatting
            answer = parsed_response.get('answer', '')
            if '%' in answer or 'percent' in answer.lower():
                confidence += 0.1
            else:
                confidence -= 0.15
        
        elif question_type == 'reference':
            # Lower confidence for reference questions as they're more complex
            confidence -= 0.05
            # Higher confidence if clear reference resolution shown
            if 'context:' in raw_response.lower() or 'refers to' in raw_response.lower():
                confidence += 0.1
        
        # Turn-based adjustments
        if turn_number > 3:
            # Slightly lower confidence for later turns due to context complexity
            confidence -= 0.02 * (turn_number - 3)
        
        # Response length heuristics
        response_length = len(parsed_response.get('reasoning_steps', []))
        if response_length < 2:
            confidence -= 0.1  # Too brief reasoning
        elif response_length > 8:
            confidence -= 0.05  # Potentially over-complicated
        
        return max(0.0, min(1.0, confidence))


class ConversationManager:
    """Handles conversation state and reference resolution with enhanced intelligence."""
    
    def __init__(self):
        self.reference_patterns = {
            'temporal': ['what about', 'and in', 'how about', 'in', 'for'],
            'calculation': ['difference', 'change', 'subtract', 'total', 'sum', 'add'],
            'reference': ['that', 'this', 'these', 'those', 'it', 'them'],
            'percentage': ['percentage', 'percent', '%', 'rate', 'ratio'],
            'comparison': ['more', 'less', 'higher', 'lower', 'greater', 'smaller'],
            'prior_value': ['previous', 'last', 'earlier', 'before', 'prior']
        }
        
        # Track conversation state for better context
        self.conversation_state = {
            'last_metric': None,
            'last_values': [],
            'calculation_results': [],
            'referenced_periods': []
        }
    
    def resolve_question_references(
        self, 
        question: str, 
        history: List[tuple[str, str]]
    ) -> str:
        """Enhanced pronoun and reference resolution in questions."""
        
        if not history:
            return question
            
        resolved = question
        q_lower = question.lower()
        
        # Update conversation state based on history
        self._update_conversation_state(history)
        
        # Handle temporal references with specific context
        if any(pattern in q_lower for pattern in self.reference_patterns['temporal']):
            context_info = self._resolve_temporal_reference(question, history)
            if context_info:
                resolved += f" [Context: {context_info}]"
        
        # Handle calculation references with value tracking
        if any(pattern in q_lower for pattern in self.reference_patterns['calculation']):
            context_info = self._resolve_calculation_reference(question, history)
            if context_info:
                resolved += f" [Context: {context_info}]"
        
        # Handle pronoun references (that, this, etc.)
        if any(pattern in q_lower for pattern in self.reference_patterns['reference']):
            context_info = self._resolve_pronoun_reference(question, history)
            if context_info:
                resolved += f" [Context: {context_info}]"
        
        # Handle comparison references
        if any(pattern in q_lower for pattern in self.reference_patterns['comparison']):
            context_info = self._resolve_comparison_reference(question, history)
            if context_info:
                resolved += f" [Context: {context_info}]"
                
        return resolved
    
    def _update_conversation_state(self, history: List[tuple[str, str]]):
        """Update internal state based on conversation history."""
        if not history:
            return
            
        # Extract metrics and values from recent questions/answers
        for question, answer in history[-3:]:  # Look at last 3 turns
            # Extract potential metric names from questions
            if any(word in question.lower() for word in ['revenue', 'income', 'cash', 'assets', 'debt']):
                # Simple metric extraction - in production would be more sophisticated
                potential_metric = self._extract_metric_from_question(question)
                if potential_metric:
                    self.conversation_state['last_metric'] = potential_metric
            
            # Extract numerical values from answers
            import re
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            if numbers:
                self.conversation_state['last_values'].extend([float(n) for n in numbers[-2:]])
                # Keep only recent values
                self.conversation_state['last_values'] = self.conversation_state['last_values'][-5:]
    
    def _resolve_temporal_reference(self, question: str, history: List[tuple[str, str]]) -> Optional[str]:
        """Resolve temporal references like 'what about in 2008?'"""
        if not history:
            return None
            
        last_question, _ = history[-1]
        
        # Extract years or periods from current question
        import re
        current_periods = re.findall(r'\b(20\d{2}|19\d{2})\b', question)
        last_periods = re.findall(r'\b(20\d{2}|19\d{2})\b', last_question)
        
        if current_periods and self.conversation_state['last_metric']:
            return f"Same metric as previous question ({self.conversation_state['last_metric']}) but for period {current_periods[0]}"
        elif 'what about' in question.lower():
            return f"Referring to same metric as previous question: '{last_question}'"
        
        return None
    
    def _resolve_calculation_reference(self, question: str, history: List[tuple[str, str]]) -> Optional[str]:
        """Resolve calculation references like 'the difference' or 'what's the change?'"""
        if len(history) < 2:
            return None
            
        if 'difference' in question.lower() or 'change' in question.lower():
            # Get the last two numerical answers
            recent_values = self.conversation_state['last_values'][-2:] if len(self.conversation_state['last_values']) >= 2 else []
            
            if len(recent_values) == 2:
                return f"Calculate difference between recent values: {recent_values[1]} - {recent_values[0]}"
            else:
                return "Calculate difference between values from recent questions"
        
        return None
    
    def _resolve_pronoun_reference(self, question: str, history: List[tuple[str, str]]) -> Optional[str]:
        """Resolve pronoun references like 'that percentage' or 'this amount'"""
        if not history:
            return None
            
        q_lower = question.lower()
        last_question, last_answer = history[-1]
        
        if 'that percentage' in q_lower or 'this percentage' in q_lower:
            return f"Refers to percentage calculation from previous context"
        elif 'that amount' in q_lower or 'this amount' in q_lower:
            return f"Refers to amount from previous answer: {last_answer}"
        elif 'it' in q_lower and len(history) >= 1:
            return f"Likely refers to result from previous question"
        
        return None
    
    def _resolve_comparison_reference(self, question: str, history: List[tuple[str, str]]) -> Optional[str]:
        """Resolve comparison references like 'is it higher than...'"""
        if not history:
            return None
            
        q_lower = question.lower()
        
        if any(comp in q_lower for comp in ['higher', 'lower', 'greater', 'less']):
            if self.conversation_state['last_values']:
                return f"Compare with recent values: {self.conversation_state['last_values'][-2:]}"
        
        return None
    
    def _extract_metric_from_question(self, question: str) -> Optional[str]:
        """Extract financial metric name from question (simplified)."""
        q_lower = question.lower()
        
        # Common financial metrics - in production would use NER or more sophisticated extraction
        metrics = {
            'revenue': ['revenue', 'sales', 'income'],
            'cash': ['cash', 'cash flow'],
            'assets': ['assets', 'total assets'],
            'debt': ['debt', 'liabilities'],
            'profit': ['profit', 'earnings', 'net income']
        }
        
        for metric_name, keywords in metrics.items():
            if any(keyword in q_lower for keyword in keywords):
                return metric_name
        
        return None
    
    def get_conversation_summary(self, history: List[tuple[str, str]]) -> Dict[str, Any]:
        """Get summary of conversation state for debugging/analysis."""
        return {
            'conversation_length': len(history),
            'last_metric': self.conversation_state['last_metric'],
            'recent_values': self.conversation_state['last_values'][-3:],
            'conversation_state': self.conversation_state
        }
