# ConvFinQA Report

## Method

### Rationale for Prompt-First Agent Architecture

My hypothesis was that GPT-4o's context capabilities could process financial documents directly with appropriate prompt design. This approach proved largely successful. Rather than building a retrieval system, I developed an agent that:

- **Structured reasoning**: Breaks down financial questions into reasoning, calculation, and final answer phases
- **Dynamic question classification**: Automatically categorizes questions as "lookup," "calculation," or "percentage" types and applies appropriate response strategies
- **Conversation context management**: Maintains the last 8 turns and resolves temporal references like "what about 2008?" or "what's the difference?"

The core idea is an adaptive prompt system that adjusts its approach based on question classification and conversation context.

### Key Technical Decisions

**Model Selection - GPT-4o over Alternative LLMs**: The choice of GPT-4o was based on pre-implementation research rather than direct testing:

- **GPT-4o vs GPT-5**: Initial testing revealed GPT-5 produced frequent empty responses with a long latency, while GPT-4o delivered consistent, faster outputs
- **GPT-4o vs Claude/Gemini**: Research indicated GPT-4o's strong performance on financial reasoning tasks and established reliability for numerical calculations
- **GPT-4o vs Open Source Models**: Research suggested that while open source models handle simple queries well, they often struggle with complex multi-step financial calculations and sustained conversation context

The decision prioritized proven reliability and financial domain capabilities based on available research and benchmarks.

**Question Classification System**: The critical breakthrough was implementing automatic question type detection. The system distinguishes between simple lookups ("what's the revenue?") and complex calculations ("what's the percentage change?"), applying tailored prompting strategies for each category.

**Financial Document Formatting**: Financial tables require careful preprocessing for optimal LLM consumption. I developed normalized formatting that significantly improved answer accuracy through better data presentation.

### Evaluation Methodology

I developed a comprehensive evaluation framework after recognizing that manual testing provides insufficient coverage for systematic assessment. Spot testing tends to favor easy cases while missing systematic failures that only emerge through broader evaluation.

The evaluation system tracks multiple complementary metrics:

- **Exact Match Rate (61.9%)**: Strict string matching between predicted and expected answers, with numerical tolerance (±0.01) to handle floating-point precision.
- **Numerical Accuracy (67.6%)**: Semantic number extraction and comparison that ignores formatting differences. For example, "15.2%" and "0.152" are considered equivalent. This metric isolates calculation accuracy from presentation issues.
- **Conversation Success Rate (30.0%)**: End-to-end evaluation of complete multi-turn conversations. A conversation is marked successful only if every question in the sequence receives a correct answer.

### Performance Results

Through systematic debugging and iterative improvement, I achieved:
- **61.9% exact match rate** (significant improvement from 0% baseline)
- **67.6% numerical accuracy** 
- **10.5s average response time**

This progression demonstrated the value of systematic evaluation in identifying and resolving fundamental implementation issues.

## Error Analysis

### Initial Implementation Challenges
The first systematic evaluation gave a 0% exact match rate despite apparently successful manual testing. This discrepancy highlighted the critical difference between selective testing and comprehensive evaluation.

Root cause analysis identified text truncation as the primary issue. Financial documents were being cut at 500 characters, often removing essential data that appeared later in the text.

### Current Limitations

**Multi-Turn Conversation Coherence**: The system achieves only 30.0% success rate on complete conversations. Complex reference resolution across multiple turns remains challenging, particularly when maintaining context through sequences of related questions.

**Percentage Calculation Formatting**: Format inconsistencies persist in percentage calculations. The agent occasionally returns "-0.343" when the expected format is "-34.3%". While most cases have been addressed, edge cases in numerical formatting continue to cause evaluation failures.

**Single-Point-of-Failure Architecture**: The system has complete dependency on GPT-4o availability and consistency. There are no fallback mechanisms or model ensemble strategies, which presents reliability risks for production deployment.

**Dataset Quality Constraints**: Analysis revealed potential ground truth inconsistencies where table data shows "95.0" but expected calculations use "0.95". Some performance limitations may stem from data quality issues rather than algorithmic shortcomings.

### Error Classification
Systematic categorization of failure modes including format inconsistencies, calculation errors, context resolution failures, and data access issues. This enables targeted debugging and optimization.

Performance Distribution: Question-type breakdown shows lookup queries (78% accuracy) significantly outperform calculation tasks (45% accuracy), indicating where algorithmic improvements should focus.

## Future Work

Given additional development time, I would focus on:

1. **Enhanced Conversation Management**: Implement sophisticated context compression and state tracking to improve multi-turn conversation success rates
2. **Robust Calculation Validation**: Add comprehensive unit handling and validation to prevent formatting errors like "1.2 billion" expanding to "1200000000" when "1.2" is expected
3. **System Reliability**: Implement model ensemble strategies, caching layers, and monitoring infrastructure for production resilience

### Key Technical Insights

**Prompt Engineering vs. Architectural Complexity**: Sophisticated prompt design with targeted examples delivered superior results compared to what I expect a hastily-implemented RAG system would achieve. This reinforces the principle that focused execution often outperforms complex but poorly-tuned architectures.

**Data Quality as Performance Foundation**: The most significant performance breakthrough came from resolving text truncation issues rather than algorithmic improvements. This experience reinforced that data preprocessing and access are as critical as model sophistication.

**Evaluation Infrastructure ROI**: Comprehensive automated evaluation initially felt like development overhead but quickly proved its value. The ability to rapidly test changes and quantify improvements was essential for systematic optimization.

## If & how you've used coding assistants or gen AI tools to help with this assignment

I used Claude (Anthropic's AI assistant) extensively throughout this project as a coding pair-programming partner. Here's how:

**Code Development**: Claude helped with implementing the core agent architecture, evaluation framework, and async processing capabilities. It was particularly valuable for writing comprehensive type hints, error handling, and following Python best practices.

**Debugging and Optimization**: When the initial evaluation showed 0% accuracy, Claude assisted in systematically diagnosing the text truncation issue and implementing fixes. It helped identify edge cases in percentage formatting and reference resolution.

**Documentation and Analysis**: Claude helped structure the evaluation metrics, generate detailed error categorizations, and write comprehensive documentation. It assisted in analyzing the results and identifying patterns in the error data.

**Code Review and Refactoring**: Claude provided suggestions for improving code organization, adding rate limiting functionality, and ensuring the codebase followed professional development standards.

**Research and Context Building**: I used ChatGPT to summarize recent papers related to the original ConvFinQA work, helping me understand developments in financial document QA and multi-hop reasoning since 2022. This research informed my decision to pursue a prompt-first approach rather than simply reimplementing the original paper's retrieval-based system with updated models.

The collaboration was transparent and iterative - I would describe problems, share code snippets, and work through solutions together. Claude's contributions were significant in achieving the 61.9% accuracy and building a production-quality evaluation framework. This report itself was structured and refined with Claude's assistance to ensure clarity and completeness.
