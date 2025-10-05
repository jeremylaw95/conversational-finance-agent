"""Sophisticated financial prompt builder for ConvFinQA."""

from typing import List, Dict, Optional, Tuple
from src.models.data_models import Document, Dialogue


class FinancialPromptBuilder:
    """Builds sophisticated prompts for financial reasoning."""
    
    def __init__(self):
        self.base_instructions = self._load_base_instructions()
        self.few_shot_examples = self._load_few_shot_examples()
    
    def build_conversation_prompt(
        self, 
        question: str,
        document: Document,
        previous_qa_pairs: List[Tuple[str, str]],  # (question, answer) pairs
        max_context_turns: int = 8
    ) -> str:
        """Main prompt building method."""
        
        # 1. Document formatting
        formatted_doc = self._format_financial_document(document)
        
        # 2. Context compression and management
        relevant_context = self._compress_conversation_context(
            previous_qa_pairs, max_context_turns
        )
        
        # 3. Few-shot example selection
        relevant_examples = self._select_relevant_examples(
            question, previous_qa_pairs
        )
        
        # 4. Build complete prompt
        prompt = f"""You are a financial analysis expert. Your task is to answer questions about financial documents using step-by-step reasoning.

{self.base_instructions}

{formatted_doc}

{relevant_examples}

{relevant_context}

CURRENT QUESTION: {question}

REASONING PROCESS:
1. **Understand the Question**: What specific information is being requested?
2. **Resolve References**: If this refers to previous context, identify what it refers to
3. **Locate Information**: Find the relevant data in the document
4. **Perform Calculations**: If needed, show step-by-step calculations
5. **Validate Answer**: Check if your answer makes sense in context

RESPONSE FORMAT:
REASONING: [Your step-by-step thinking process]
CALCULATION: [If applicable, show calculation in format: operation(arg1, arg2)]
ANSWER: [Final numerical answer only - no units, no dollar signs, no "billion/million". For percentages, round to 1 decimal place (e.g., 14.1%)]
"""
        return prompt
    
    def _format_financial_document(self, document: Document) -> str:
        """Format financial document for optimal LLM consumption."""
        
        # Format table as clean structure
        table_str = "FINANCIAL DATA TABLE:\n"
        if document.table:
            # Get headers (years/periods)
            headers = list(document.table.keys())
            
            # Get all metrics
            all_metrics = set()
            for period_data in document.table.values():
                if isinstance(period_data, dict):
                    all_metrics.update(period_data.keys())
            
            # Create formatted table
            table_str += f"{'Metric':<40} | " + " | ".join(f"{h:>15}" for h in headers) + "\n"
            table_str += "-" * (40 + len(headers) * 18) + "\n"
            
            for metric in sorted(all_metrics):
                row = f"{metric:<40} | "
                for header in headers:
                    if isinstance(document.table[header], dict):
                        value = document.table[header].get(metric, "N/A")
                        if isinstance(value, (int, float)):
                            row += f"{value:>15,.1f} | "
                        else:
                            row += f"{str(value):>15} | "
                    else:
                        row += f"{'N/A':>15} | "
                table_str += row + "\n"
        
        # Truncate long text sections, but keep more for financial data
        pre_text_display = document.pre_text[:3000] + "..." if len(document.pre_text) > 3000 else document.pre_text
        post_text_display = document.post_text[:3000] + "..." if len(document.post_text) > 3000 else document.post_text
        
        return f"""FINANCIAL DOCUMENT:
PRE-TEXT: {pre_text_display}

{table_str}

POST-TEXT: {post_text_display}"""

    def _compress_conversation_context(
        self, 
        previous_qa_pairs: List[Tuple[str, str]], 
        max_turns: int
    ) -> str:
        """Smart context compression inspired by MEM1."""
        
        if not previous_qa_pairs:
            return "CONVERSATION HISTORY: (This is the first question)"
        
        # Take most recent turns, but include key reference points
        recent_pairs = previous_qa_pairs[-max_turns:]
        
        context = "CONVERSATION HISTORY:\n"
        for i, (question, answer) in enumerate(recent_pairs, 1):
            context += f"Q{i}: {question}\n"
            context += f"A{i}: {answer}\n"
        
        # Add reference resolution hints
        context += "\nREFERENCE RESOLUTION GUIDE:\n"
        context += "- 'what about [year]?' -> same metric as previous question, different year\n"
        context += "- 'the difference' -> typically means later value minus earlier value\n"
        context += "- 'the change over the year' -> later year minus earlier year (e.g., 2017 - 2016)\n"
        context += "- 'that percentage' -> refers to percentage calculation pattern from context\n"
        context += "- 'these amounts' -> refers to values mentioned in recent questions\n"
        
        return context
    
    def _select_relevant_examples(
        self, 
        question: str, 
        previous_qa_pairs: List[Tuple[str, str]]
    ) -> str:
        """Dynamic few-shot example selection (DoubleDipper technique)."""
        
        # Determine question type
        question_type = self._classify_question_type(question, previous_qa_pairs)
        
        examples = {
            "lookup": """EXAMPLE - Direct Lookup:
Q: What is the net cash from operating activities in 2009?
REASONING: Looking at the financial data table under "Year ended June 30, 2009", I can find "net cash from operating activities" with value 206588.0
CALCULATION: N/A (direct lookup)
ANSWER: 206588""",
            
            "reference": """EXAMPLE - Reference Resolution:
Previous: Q1: What is the net cash from operating activities in 2009? A1: 206588
Current: Q2: what about in 2008?
REASONING: "what about in 2008?" refers to the same metric as Q1 (net cash from operating activities) but for year 2008. Looking at the table under "2008" column, I find 181001.0
CALCULATION: N/A (direct lookup)
ANSWER: 181001""",
            
            "calculation": """EXAMPLE - Multi-step Calculation:
Previous: Q1: 206588, Q2: 181001
Current: Q3: what is the difference?
REASONING: "The difference" refers to the difference between values from Q1 and Q2. Financial analysis typically uses newer period minus older period.
CALCULATION: subtract(206588, 181001)
ANSWER: 25587""",
            
            "percentage": """EXAMPLE - Percentage Calculation:
Previous: Q3 result: subtract(206588, 181001) = 25587
Current: Q4: what percentage change does this represent?
REASONING: Percentage change = (difference / base_value) * 100. Base value is the earlier year (2008 = 181001). The difference is already calculated as 25587. I need to calculate 25587 / 181001 * 100.
CALCULATION: divide(25587, 181001), multiply(#0, 100)
ANSWER: 14.1%

EXAMPLE - Portion/Ratio Calculation:
Question: What portion of total liabilities is current liabilities?  
Context: Current liabilities: 50,000; Total liabilities: 200,000
REASONING: Portion means percentage. I need to calculate (current liabilities / total liabilities) * 100 = (50,000 / 200,000) * 100.
CALCULATION: divide(50000, 200000), multiply(#0, 100)  
ANSWER: 25.0%

EXAMPLE - Percentage Rounding:
Question: What portion is 38480 of 2200900?
REASONING: Calculate (38480 / 2200900) * 100 = 1.7478%. Round to 1 decimal place.
CALCULATION: divide(38480, 2200900), multiply(#0, 100)
ANSWER: 1.7%"""
        }
        
        return f"REASONING EXAMPLE:\n{examples.get(question_type, examples['lookup'])}\n"
    
    def _classify_question_type(self, question: str, previous_qa_pairs: List[Tuple[str, str]]) -> str:
        """Classify question type for example selection."""
        q_lower = question.lower()
        
        # Check percentage first, as percentage questions often contain "change"
        if any(pct in q_lower for pct in ["percentage", "percent", "%", "portion", "proportion", "ratio", "rate", "what part"]):
            return "percentage"
        elif any(ref in q_lower for ref in ["what about", "and in", "how about"]):
            return "reference"
        elif any(calc in q_lower for calc in ["difference", "change", "subtract"]):
            return "calculation"  
        else:
            return "lookup"
    
    def _load_base_instructions(self) -> str:
        """Load core instructions for financial analysis."""
        return """CORE INSTRUCTIONS:
- You are analyzing financial documents with tables and text
- Answer questions step-by-step with clear reasoning
- For calculations, use the format: operation(arg1, arg2)
- Use #0, #1, etc. to reference intermediate calculation results
- Be precise with numerical values from the document
- If context refers to previous questions, resolve those references clearly
- Always check your work for accuracy
- Financial calculations should be precise to avoid rounding errors"""
    
    def _load_few_shot_examples(self) -> List[str]:
        """Load examples from dataset analysis."""
        # This would load examples from dataset analysis in full implementation
        return []
    
    def add_self_correction_layer(self, base_prompt: str, question_type: str) -> str:
        """Add specific self-correction for different question types."""
        
        corrections = {
            'lookup': """
SELF-CHECK for Direct Lookup:
- Did I find the exact metric requested?
- Did I use the correct year/period?
- Is the number exactly as shown in the table?""",
            
            'calculation': """
SELF-CHECK for Calculations:
- Are my calculation steps in the correct order?
- Did I use the right format: operation(arg1, arg2)?
- Do the numbers match what was discussed earlier?""",
            
            'reference': """
SELF-CHECK for Reference Questions:
- What exactly does this question refer to from previous context?
- Am I using the same metric/concept as referenced?
- Did I identify the correct time period?""",
            
            'percentage': """
SELF-CHECK for Percentage Calculations:
- Did I identify the correct base value for the percentage?
- Is my calculation formula correct: (part / total) * 100 or (difference / base) * 100?
- Did I format the answer with a % symbol?
- Are my reasoning steps clear and logical?
- Does the percentage value make sense given the context (typically 0-100%)?
- Am I calculating a percentage rather than just extracting raw numbers?"""
        }
        
        return base_prompt + corrections.get(question_type, corrections['lookup'])

