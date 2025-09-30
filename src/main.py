"""
Main typer app for ConvFinQA
"""

import typer
import json
import logging
import asyncio
from pathlib import Path
from rich import print as rich_print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import our ConvFinQA components
from src.agents.conv_finqa_agent import ConvFinQAAgent
from src.utils.data_loader import DataLoader
from src.logger import get_logger
from src.config import ConvFinQAConfig
from src.evaluation.evaluator import ConvFinQAEvaluator
from src.evaluation.metrics import MetricsReporter

console = Console()
logger = get_logger(__name__)

app = typer.Typer(
    name="main",
    help="ConvFinQA Agent with evaluation capabilities",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
) -> None:
    """Interactive chat with agent about a specific record"""
    
    # Load data
    loader = DataLoader()
    record = loader.load_record_by_id(record_id)
    
    if not record:
        console.print(f"[red]Record {record_id} not found[/red]")
        return
    
    try:
        agent = ConvFinQAAgent(model_name=ConvFinQAConfig.DEFAULT_MODEL)
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        console.print("[yellow]Make sure OPEN_AI_API_KEY is set in your environment[/yellow]")
        console.print(f"[dim]Looking for API key in: OPENAI_API_KEY or OPEN_AI_API_KEY[/dim]")
        return
    
    previous_qa_pairs = []
    
    console.print(Panel(f"[green]Loaded record: {record.id}[/green]", title="ConvFinQA Agent"))
    console.print("[yellow]Type 'exit' or 'quit' to end the conversation[/yellow]")
    console.print("[cyan]Example questions from dataset:[/cyan]")
    for i, q in enumerate(record.dialogue.conv_questions[:2], 1):
        console.print(f"  {i}. {q}")
    console.print()
    
    while True:
        try:
            question = typer.prompt("Question")
        except (EOFError, KeyboardInterrupt):
            break
        
        if question.strip().lower() in ["exit", "quit"]:
            break
        
        # Process question
        with console.status("[bold blue]Thinking..."):
            response = agent.process_single_question(
                question=question,
                document=record.doc,
                previous_qa_pairs=previous_qa_pairs,
                turn_number=len(previous_qa_pairs) + 1
            )
        
        previous_qa_pairs.append((question, response.answer))
        
        # Display response
        console.print(f"[bold green]Answer:[/bold green] {response.answer}")
        console.print(f"[bold blue]Reasoning:[/bold blue]")
        for step in response.reasoning_steps:
            console.print(f"  • {step}")
        console.print(f"[dim]Confidence: {response.confidence:.2f} | Time: {response.processing_time:.2f}s[/dim]")
        console.print()  # Add spacing between turns


@app.command()
def dataset_info() -> None:
    """Display dataset information and statistics"""
    loader = DataLoader()
    
    try:
        stats = loader.get_dataset_statistics()
        
        console.print(Panel("[bold]ConvFinQA Dataset Information[/bold]", style="blue"))
        
        for split_name, split_stats in stats.items():
            table = Table(title=f"{split_name.title()} Split")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Total Records", str(split_stats["total_records"]))
            if "avg_dialogue_turns" in split_stats:
                table.add_row("Avg Dialogue Turns", f"{split_stats['avg_dialogue_turns']:.1f}")
                table.add_row("Max Dialogue Turns", str(split_stats["max_dialogue_turns"]))
                table.add_row("Min Dialogue Turns", str(split_stats["min_dialogue_turns"]))
            
            # Question type statistics
            qt = split_stats["question_types"]
            table.add_row("Type2 Questions", str(qt["type2_questions"]))
            table.add_row("Duplicate Columns", str(qt["duplicate_columns"]))
            table.add_row("Non-numeric Values", str(qt["non_numeric_values"]))
            
            console.print(table)
            console.print()
            
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")


@app.command()
def validate_dataset() -> None:
    """Validate dataset structure and report issues"""
    loader = DataLoader()
    
    with console.status("[bold blue]Validating dataset..."):
        results = loader.validate_dataset()
    
    console.print(Panel("[bold]Dataset Validation Results[/bold]", style="green"))
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")
    
    table.add_row("Valid Records", str(results["valid_records"]))
    table.add_row("Invalid Records", str(results["invalid_records"]))
    table.add_row("Total Errors", str(len(results["errors"])))
    
    console.print(table)
    
    if results["errors"]:
        console.print("\n[bold red]Errors Found:[/bold red]")
        for error in results["errors"][:10]:  # Show first 10 errors
            console.print(f"  • {error}")
        if len(results["errors"]) > 10:
            console.print(f"  ... and {len(results['errors']) - 10} more errors")


@app.command()
def test_agent(
    record_id: str = typer.Option(None, help="Specific record ID to test"),
    num_questions: int = typer.Option(2, help="Number of questions to test")
) -> None:
    """Test agent on dataset questions"""
    loader = DataLoader()
    
    if record_id:
        record = loader.load_record_by_id(record_id)
        if not record:
            console.print(f"[red]Record {record_id} not found[/red]")
            return
        test_records = [record]
    else:
        test_records = loader.load_test_samples(1)
        
    if not test_records:
        console.print("[red]No test records found[/red]")
        return
        
    try:
        agent = ConvFinQAAgent(model_name=ConvFinQAConfig.DEFAULT_MODEL)
    except Exception as e:
        console.print(f"[red]Error initializing agent: {e}[/red]")
        console.print("[yellow]Make sure OPEN_AI_API_KEY is set in your environment[/yellow]")
        console.print(f"[dim]Looking for API key in: OPENAI_API_KEY or OPEN_AI_API_KEY[/dim]")
        return
    
    record = test_records[0]
    console.print(Panel(f"[bold]Testing Agent on Record: {record.id}[/bold]", style="blue"))
    
    # Test first few questions from the dataset
    questions_to_test = record.dialogue.conv_questions[:num_questions]
    expected_answers = record.dialogue.conv_answers[:num_questions]
    
    previous_qa_pairs = []
    
    for i, (question, expected) in enumerate(zip(questions_to_test, expected_answers), 1):
        console.print(f"\n[bold cyan]Question {i}:[/bold cyan] {question}")
        console.print(f"[bold yellow]Expected:[/bold yellow] {expected}")
        
        with console.status(f"[bold blue]Processing question {i}..."):
            response = agent.process_single_question(
                question=question,
                document=record.doc,
                previous_qa_pairs=previous_qa_pairs,
                turn_number=i
            )
        
        console.print(f"[bold green]Agent Answer:[/bold green] {response.answer}")
        console.print(f"[bold blue]Reasoning:[/bold blue]")
        for step in response.reasoning_steps:
            console.print(f"  • {step}")
        
        # Simple accuracy check
        exact_match = response.answer.strip().lower() == expected.strip().lower()
        accuracy_status = "[green]EXACT MATCH[/green]" if exact_match else "[red]NO MATCH[/red]"
        console.print(f"[bold]Accuracy:[/bold] {accuracy_status}")
        console.print(f"[dim]Confidence: {response.confidence:.2f} | Time: {response.processing_time:.2f}s[/dim]")
        
        previous_qa_pairs.append((question, response.answer))


@app.command()
def evaluate(
    model: str = typer.Option("gpt-4o", help="LLM model to use"),
    num_samples: int = typer.Option(10, help="Number of test samples"),
    output: str = typer.Option("evaluation_results.json", help="Output file for results"),
    split: str = typer.Option("dev", help="Dataset split to use (train/dev)"),
    verbose: bool = typer.Option(False, help="Verbose output"),
    async_mode: bool = typer.Option(True, help="Use async processing for faster evaluation")
) -> None:
    """Run comprehensive evaluation on test dataset"""
    
    console.print(Panel(f"[bold]ConvFinQA Evaluation Framework[/bold]", style="blue"))
    console.print(f"Model: [cyan]{model}[/cyan] | Samples: [cyan]{num_samples}[/cyan] | Split: [cyan]{split}[/cyan]")
    
    try:
        # Initialize components
        loader = DataLoader()
        agent = ConvFinQAAgent(model_name=model)
        evaluator = ConvFinQAEvaluator()
        reporter = MetricsReporter()
        
        console.print("\n[blue]Loading test data...[/blue]")
        test_data = loader.load_test_samples(num_samples=num_samples, split=split)
        
        if not test_data:
            console.print("[red]No test data loaded[/red]")
            return
        
        console.print(f"[green]Loaded {len(test_data)} conversations[/green]")
        
        # Run evaluation (async or sync based on flag)
        all_metrics = []
        successful_evaluations = 0
        
        if async_mode:
            console.print("[blue]Using async processing for faster evaluation...[/blue]")
        
        async def process_conversation_async(sample, i):
            """Process a single conversation asynchronously."""
            try:
                # Run agent on conversation using async method
                responses = await agent.process_conversation_async(sample)
                
                # Evaluate responses
                metrics = evaluator.evaluate_full_conversation(
                    predicted_responses=responses,
                    ground_truth_answers=sample.dialogue.conv_answers,
                    ground_truth_programs=sample.dialogue.turn_program,
                    conversation_questions=sample.dialogue.conv_questions
                )
                
                if verbose:
                    console.print(f"   [green]OK[/green] {sample.id}: {metrics.exact_match_rate:.1%} accuracy")
                
                return metrics, sample.id
                
            except Exception as e:
                console.print(f"   [red]ERROR[/red] processing {sample.id}: {str(e)}")
                logger.error(f"Evaluation error for {sample.id}: {e}")
                return None, sample.id
        
        async def run_all_evaluations():
            """Run all evaluations concurrently with controlled concurrency."""
            # Limit concurrent requests to avoid overwhelming the API
            semaphore = asyncio.Semaphore(5)  # Max 5 concurrent conversations
            
            async def process_with_semaphore(sample, i):
                async with semaphore:
                    return await process_conversation_async(sample, i)
            
            # Create tasks for all conversations with semaphore control
            tasks = [process_with_semaphore(sample, i) for i, sample in enumerate(test_data)]
            
            # Run with progress updates
            completed = 0
            for coro in asyncio.as_completed(tasks):
                result, sample_id = await coro
                completed += 1
                console.print(f"   Progress: {completed}/{len(test_data)} conversations completed")
                
                if result:
                    all_metrics.append(result)
                    
            return len([m for m in all_metrics if m is not None])
        
        if async_mode:
            with console.status("[bold green]Running async evaluations..."):
                successful_evaluations = asyncio.run(run_all_evaluations())
        else:
            # Fallback to original sync processing
            console.print("[blue]Using synchronous processing...[/blue]")
            with console.status("[bold green]Running evaluations...") as status:
                for i, sample in enumerate(test_data):
                    status.update(f"[bold green]Processing conversation {i+1}/{len(test_data)}: {sample.id[:30]}...")
                    
                    try:
                        # Run agent on conversation
                        responses = agent.process_conversation(sample)
                        
                        # Evaluate responses
                        metrics = evaluator.evaluate_full_conversation(
                            predicted_responses=responses,
                            ground_truth_answers=sample.dialogue.conv_answers,
                            ground_truth_programs=sample.dialogue.turn_program,
                            conversation_questions=sample.dialogue.conv_questions
                        )
                        
                        all_metrics.append(metrics)
                        successful_evaluations += 1
                        
                        if verbose:
                            console.print(f"   [green]OK[/green] {sample.id}: {metrics.exact_match_rate:.1%} accuracy")
                            
                    except Exception as e:
                        console.print(f"   [red]ERROR[/red] processing {sample.id}: {str(e)}")
                        logger.error(f"Evaluation error for {sample.id}: {e}")
        
        if not all_metrics:
            console.print("[red]No successful evaluations completed[/red]")
            return
        
        # Aggregate results
        console.print(f"\n[blue]Aggregating results from {successful_evaluations} conversations...[/blue]")
        aggregated_metrics = evaluator.aggregate_metrics(all_metrics)
        
        # Generate comprehensive report
        report = reporter.generate_detailed_report(
            metrics=aggregated_metrics,
            test_conversations=[sample.model_dump() for sample in test_data[:5]],  # Sample for context
            model_name=model,
            additional_context={
                "num_samples": num_samples,
                "split": split,
                "successful_evaluations": successful_evaluations,
                "total_attempted": len(test_data)
            }
        )
        
        # Save results
        output_path = Path(output)
        reporter.save_report(report, str(output_path))
        
        # Display summary
        reporter.print_summary(aggregated_metrics)
        
        # Show key insights
        console.print(f"\n[bold]Key Insights:[/bold]")
        if report["recommendations"]:
            for rec in report["recommendations"][:3]:  # Show top 3
                console.print(f"   • {rec}")
        
        console.print(f"\n[green]Evaluation complete![/green] Full report saved to [cyan]{output_path}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]ERROR: Evaluation failed: {str(e)}[/red]")
        logger.error(f"Evaluation command failed: {e}")
        raise typer.Exit(1)


@app.command()
def benchmark(
    record_id: str = typer.Argument(..., help="Specific record ID to benchmark"),
    model: str = typer.Option("gpt-4o", help="LLM model to use"),
    runs: int = typer.Option(3, help="Number of runs for statistical reliability")
) -> None:
    """Benchmark agent performance on a specific conversation"""
    
    console.print(Panel(f"[bold]ConvFinQA Benchmark[/bold]", style="yellow"))
    console.print(f"Record: [cyan]{record_id}[/cyan] | Model: [cyan]{model}[/cyan] | Runs: [cyan]{runs}[/cyan]")
    
    try:
        # Load specific record
        loader = DataLoader()
        record = loader.load_record_by_id(record_id)
        
        if not record:
            console.print(f"[red]ERROR: Record '{record_id}' not found[/red]")
            raise typer.Exit(1)
        
        agent = ConvFinQAAgent(model_name=model)
        evaluator = ConvFinQAEvaluator()
        
        console.print(f"\n[blue]Running {runs} evaluation runs...[/blue]")
        
        all_run_metrics = []
        
        for run in range(runs):
            console.print(f"   [yellow]Run {run + 1}/{runs}[/yellow]")
            
            # Run agent
            responses = agent.process_conversation(record)
            
            # Evaluate
            metrics = evaluator.evaluate_full_conversation(
                predicted_responses=responses,
                ground_truth_answers=record.dialogue.conv_answers,
                ground_truth_programs=record.dialogue.turn_program,
                conversation_questions=record.dialogue.conv_questions
            )
            
            all_run_metrics.append(metrics)
            console.print(f"      Accuracy: {metrics.exact_match_rate:.1%} | Time: {metrics.avg_response_time:.1f}s")
        
        # Calculate statistics
        accuracies = [m.exact_match_rate for m in all_run_metrics]
        times = [m.avg_response_time for m in all_run_metrics]
        
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_time = sum(times) / len(times)
        
        console.print(f"\n[bold]Benchmark Results:[/bold]")
        console.print(f"   Average Accuracy: {avg_accuracy:.1%}")
        console.print(f"   Average Response Time: {avg_time:.2f}s")
        console.print(f"   Consistency: {min(accuracies):.1%} - {max(accuracies):.1%}")
        
        # Show detailed breakdown
        if all_run_metrics[0].error_breakdown:
            console.print(f"\n[bold]Error Analysis:[/bold]")
            for error_type, count in all_run_metrics[0].error_breakdown.items():
                if count > 0:
                    console.print(f"   • {error_type}: {count}")
        
    except Exception as e:
        console.print(f"[red]ERROR: Benchmark failed: {str(e)}[/red]")
        logger.error(f"Benchmark command failed: {e}")
        raise typer.Exit(1)


@app.command()
def myfunc() -> None:
    """Demo function showing ConvFinQA Agent capabilities"""
    console.print(Panel("[bold]ConvFinQA Agent - Demo Mode[/bold]", style="green"))
    console.print("Available commands:")
    console.print("  • [cyan]dataset-info[/cyan] - View dataset statistics")
    console.print("  • [cyan]validate-dataset[/cyan] - Validate dataset structure")
    console.print("  • [cyan]test-agent[/cyan] - Test agent on sample questions")
    console.print("  • [cyan]chat <record-id>[/cyan] - Interactive chat with a record")
    console.print("  • [cyan]evaluate[/cyan] - Run comprehensive evaluation framework")
    console.print("  • [cyan]benchmark <record-id>[/cyan] - Benchmark specific conversation")
    console.print("\n[yellow]Example:[/yellow] python -m src.main evaluate --num-samples 5")


if __name__ == "__main__":
    app()
