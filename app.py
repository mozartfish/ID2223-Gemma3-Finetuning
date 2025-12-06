import os
import subprocess
import gradio as gr
from huggingface_hub import hf_hub_download

# Model configuration
REPO_ID = "mozartfish/Gemma3-FineTome30K-data-centric"
FILENAME = "gemma-3-data-centric-q8_0.gguf"

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir="./models"
)

# Binary paths
LLAMA_BIN = "/llama.cpp/build/bin/llama-cli"
LLAMA_PERPLEXITY_BIN = "/llama.cpp/build/bin/llama-perplexity"


def run_llama(prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
    cmd = [
        LLAMA_BIN,
        "-m", model_path,
        "--simple-io",
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--top-p", str(top_p),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.stdout.strip():
        response = result.stdout.strip()
        
        # Clean up common formatting artifacts - improved cleaning
        lines = response.split('\n')
        cleaned_lines = []
        skip_until_model = True  # Skip everything until we see "model" label
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip lines that are just labels
            if line_stripped.lower() in ['user', 'model', 'assistant', 'eof by user', 'user:', 'assistant:', 'model:']:
                if line_stripped.lower() in ['model', 'model:']:
                    skip_until_model = False
                continue
            
            if line_stripped.startswith('>') and 'eof' in line_stripped.lower():
                continue
            
            if skip_until_model:
                continue
            
            if not cleaned_lines and not line_stripped:
                continue
            
            cleaned_lines.append(line)
        
        while cleaned_lines and (
            cleaned_lines[-1].strip().lower().startswith('eof') or 
            cleaned_lines[-1].strip().startswith('>') or
            not cleaned_lines[-1].strip()
        ):
            cleaned_lines.pop()
        
        cleaned_response = '\n'.join(cleaned_lines).strip()
        
        if not cleaned_response:
            return response
        
        return cleaned_response
    
    return f"[Model produced no output. Exit code: {result.returncode}\n{result.stderr}]"


# -------------------------------------------------------
# Chatbot for ChatInterface
def respond(message, history, temperature, top_p, max_tokens):
    prompt = ""
    for turn in history:
        role = turn["role"]
        content = turn["content"]
        prompt += f"{role.capitalize()}: {content}\n"

    prompt += f"User: {message}\nAssistant:"
    return run_llama(prompt, temperature, top_p, max_tokens)


# Text Rewriting Service
def rewrite_text(text, style, temperature, top_p, max_tokens):
    style_prompts = {
        "Gen-Z": "Rewrite the following text using Gen-Z slang and style. Use terms like 'no cap', 'fr fr', 'lowkey', 'highkey', 'slaps', 'vibe', etc. Make it casual, fun, and relatable.\n\n",
        "Formal Corporate": "Rewrite the following text in formal corporate business language. Be professional, concise, and maintain a polished tone. Avoid slang and casual expressions.\n\n",
        "Academic": "Rewrite the following text in academic style. Use scholarly language, precise terminology, and maintain intellectual rigor. Be thorough and educational.\n\n",
        "Humorous": "Rewrite the following text in a humorous and witty style. Add clever jokes, wordplay, and make it entertaining while keeping the core message.\n\n",
        "Teacher Mode": "Rewrite the following text in teacher mode. Break down concepts into simple steps, use analogies, and provide clear explanations. Be patient and pedagogical.\n\n"
    }
    
    style_instruction = style_prompts.get(style, style_prompts["Formal Corporate"])
    prompt = f"{style_instruction}Original text:\n{text}\n\nRewritten text:"
    
    return run_llama(prompt, temperature, top_p, max_tokens)


# Batch Evaluation - Perplexity Computation
def compute_ppl_llama_cpp(gguf_path, text):
    import tempfile
    import re
    import os

    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt", encoding="utf-8") as f:
        f.write(text.strip())
        temp_path = f.name

    if os.path.exists(LLAMA_PERPLEXITY_BIN):
        cmd = [
            LLAMA_PERPLEXITY_BIN,
            "-m", gguf_path,
            "-f", temp_path,
            "-c", "512",
            "--chunks", "1"
        ]
    else:
        os.unlink(temp_path)
        return "[ERROR] llama-perplexity binary not found. Please rebuild llama.cpp with perplexity support."

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return "[ERROR] Perplexity computation timed out (>120s)"
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return f"[ERROR] Exception during perplexity computation: {str(e)}"
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

    stdout = result.stdout
    stderr = result.stderr
    combined_output = stdout + "\n" + stderr

    patterns = [
        r"Final\s+estimate:\s+PPL\s*=\s*([\d\.]+)",
        r"perplexity\s*[:\s=]+([\d\.]+)",
        r"ppl\s*[:\s=]+([\d\.]+)",
        r"PPL\s*=\s*([\d\.]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, combined_output, re.IGNORECASE)
        if match:
            try:
                ppl_value = float(match.group(1))
                return ppl_value
            except ValueError:
                continue

    if "you need at least" in combined_output and "tokens to evaluate perplexity" in combined_output:
        token_match = re.search(r"need at least (\d+) tokens.*?tokenizes to only (\d+) tokens", combined_output)
        if token_match:
            required = token_match.group(1)
            provided = token_match.group(2)
            return f"[ERROR] Text too short: needs {required}+ tokens, got {provided} tokens. Please provide more text for evaluation."
        return "[ERROR] Text is too short to compute perplexity. Please provide more text (at least 1024 tokens)."
    
    error_msg = f"[ERROR] Could not parse perplexity from output."
    if result.returncode != 0:
        error_msg += f" Exit code: {result.returncode}"
    error_msg += f"\nOutput length: STDOUT={len(stdout)}, STDERR={len(stderr)}"
    error_msg += f"\nLast 500 chars: {combined_output[-500:]}"
    return error_msg



def batch_eval(text):
    import matplotlib.pyplot as plt

    models = {
        "Baseline (10K)": hf_hub_download(
            repo_id="mozartfish/Gemma3-FineTome10K-Baseline",
            filename="gemma-3-baseline-q8_0.gguf",
            local_dir="./models/baseline"
        ),
        "Model-Centric (10K)": hf_hub_download(
            repo_id="mozartfish/Gemma3-FineTome10K-model-centric",
            filename="gemma-3-model-centric-q8_0.gguf",
            local_dir="./models/model-centric"
        ),
        "Data-Centric (30K)": hf_hub_download(
            repo_id="mozartfish/Gemma3-FineTome30K-data-centric",
            filename="gemma-3-data-centric-q8_0.gguf",
            local_dir="./models/data-centric"
        ),
    }

    results = []
    for name, path in models.items():
        ppl = compute_ppl_llama_cpp(path, text)
        results.append([name, ppl])

    names = [r[0] for r in results]
    ppls = [r[1] if isinstance(r[1], (int, float)) else 0 for r in results]
    has_valid_data = any(isinstance(r[1], (int, float)) for r in results)

    fig, ax = plt.subplots(figsize=(6, 4))
    
    if has_valid_data:
        bars = ax.bar(names, ppls)
        ax.set_ylabel("Perplexity ↓")
        ax.set_title("Model Comparison (llama.cpp Perplexity)")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right")
        plt.tight_layout()
        
        for i, (name, ppl) in enumerate(results):
            if not isinstance(ppl, (int, float)):
                ax.text(i, 0, "ERROR", ha='center', va='bottom', color='red', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "All models failed to compute perplexity.\nCheck logs for details.", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    return results, fig


# Evaluation Methods

def get_model_paths():
    return {
        "Baseline (10K)": hf_hub_download(
            repo_id="mozartfish/Gemma3-FineTome10K-Baseline",
            filename="gemma-3-baseline-q8_0.gguf",
            local_dir="./models/baseline"
        ),
        "Model-Centric (10K)": hf_hub_download(
            repo_id="mozartfish/Gemma3-FineTome10K-model-centric",
            filename="gemma-3-model-centric-q8_0.gguf",
            local_dir="./models/model-centric"
        ),
        "Data-Centric (30K)": hf_hub_download(
            repo_id="mozartfish/Gemma3-FineTome30K-data-centric",
            filename="gemma-3-data-centric-q8_0.gguf",
            local_dir="./models/data-centric"
        ),
    }


def run_single_generation(model_path, prompt, max_tokens=256, temperature=0.7, top_p=0.95):
    import time
    
    cmd = [
        LLAMA_BIN,
        "-m", model_path,
        "--simple-io",
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temperature),
        "--top-p", str(top_p),
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    duration = time.time() - start
    
    if result.stdout.strip():
        response = result.stdout.strip()
        lines = response.split('\n')
        cleaned_lines = []
        skip_until_model = True
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.lower() in ['user', 'model', 'assistant', 'eof by user', 'user:', 'assistant:', 'model:']:
                if line_stripped.lower() in ['model', 'model:']:
                    skip_until_model = False
                continue
            
            if line_stripped.startswith('>') and 'eof' in line_stripped.lower():
                continue
            
            if skip_until_model:
                continue
            
            if not cleaned_lines and not line_stripped:
                continue
            
            cleaned_lines.append(line)
        
        while cleaned_lines and (
            cleaned_lines[-1].strip().lower().startswith('eof') or 
            cleaned_lines[-1].strip().startswith('>') or
            not cleaned_lines[-1].strip()
        ):
            cleaned_lines.pop()
        
        cleaned_response = '\n'.join(cleaned_lines).strip()
        response = cleaned_response if cleaned_response else response
    else:
        response = "[No output]"
    
    return response, duration


def speed_benchmark(num_tokens):
    import matplotlib.pyplot as plt
    
    models = get_model_paths()
    test_prompt = "Explain the concept of machine learning in simple terms:"
    
    results = []
    speeds = []
    names = []
    
    for name, path in models.items():
        try:
            _, duration = run_single_generation(path, test_prompt, max_tokens=num_tokens, temperature=0.7, top_p=0.95)
            tokens_per_sec = num_tokens / duration if duration > 0 else 0
            
            results.append([name, f"{duration:.2f}s", f"{tokens_per_sec:.1f}"])
            speeds.append(tokens_per_sec)
            names.append(name)
        except Exception as e:
            results.append([name, "ERROR", "0"])
            speeds.append(0)
            names.append(name)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, speeds, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_ylabel("Tokens per Second")
    ax.set_title("Speed Benchmark: Tokens/Second")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    
    for i, (bar, speed) in enumerate(zip(bars, speeds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speed:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    return results, fig


def few_shot_tasks():
    models = get_model_paths()
    
    tasks = {
        "Math": "Calculate: What is 15% of 240? Show your work.",
        "Reasoning": "If all roses are flowers and some flowers fade quickly, can we conclude that all roses fade quickly? Explain why or why not.",
        "Summarization": "Summarize this in one sentence: Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data and improve their performance over time without being explicitly programmed.",
        "Code": "Write a Python function that reverses a string.",
    }
    
    model_results = {
        "Baseline (10K)": [],
        "Model-Centric (10K)": [],
        "Data-Centric (30K)": []
    }
    
    for task_name, prompt in tasks.items():
        for model_name, model_path in models.items():
            try:
                response, duration = run_single_generation(model_path, prompt, max_tokens=150, temperature=0.7, top_p=0.95)
                model_results[model_name].append(f"**Task: {task_name}** ({duration:.1f}s)\n{response}\n")
            except Exception as e:
                model_results[model_name].append(f"**Task: {task_name}**\n[ERROR] {str(e)}\n")
    
    baseline_output = "\n\n" + "="*50 + "\n\n".join(model_results["Baseline (10K)"])
    model_centric_output = "\n\n" + "="*50 + "\n\n".join(model_results["Model-Centric (10K)"])
    data_centric_output = "\n\n" + "="*50 + "\n\n".join(model_results["Data-Centric (30K)"])
    
    return baseline_output, model_centric_output, data_centric_output


# Gradio UI
with gr.Blocks(title="Gemma-3 Fine-Tuned LLM") as demo:

    gr.Markdown("# Gemma-3 Fine-Tuned LLM\nChat + Prompt Boost + Batch Evaluation")

    temperature = gr.Slider(0, 2, value=0.7, label="Temperature")
    top_p = gr.Slider(0, 1, value=0.95, label="Top-p")
    max_tokens = gr.Slider(32, 1024, value=256, step=1, label="Max Tokens")

    with gr.Tab("Chatbot"):
        gr.Markdown("### Chat with your fine-tuned model")
        gr.ChatInterface(
            fn=respond,
            additional_inputs=[temperature, top_p, max_tokens]
        )

    with gr.Tab("Text Rewriter"):
        gr.Markdown("### Rewrite Text in Different Styles")
        gr.Markdown("Transform your text using different writing styles and tones!")
        
        rewrite_input = gr.Textbox(
            lines=8,
            label="Input Text",
            value="Artificial intelligence is changing how we work and live. Machine learning algorithms can now process large amounts of data quickly. This technology helps businesses make better decisions and improves efficiency in many industries.",
            placeholder="Enter text to rewrite..."
        )
        
        style_selector = gr.Radio(
            choices=["Gen-Z", "Formal Corporate", "Academic", "Humorous", "Teacher Mode"],
            label="Rewriting Style",
            value="Gen-Z",
            info="Choose the style for rewriting"
        )
        
        rewrite_button = gr.Button("Rewrite Text", variant="primary")
        rewrite_output = gr.Textbox(lines=10, label="Rewritten Output")
        
        rewrite_button.click(
            rewrite_text,
            inputs=[rewrite_input, style_selector, temperature, top_p, max_tokens],
            outputs=rewrite_output,
        )

    with gr.Tab("Batch Model Comparison"):
        gr.Markdown("### Compare All Models at Once Using Perplexity")
        gr.Markdown("""
        **Note:** Perplexity computation requires at least 1024 tokens of text. 
        If you provide less text, you'll see an error. Use the sample text below or paste your own longer text.
        """)
        
        # Default sample text (long enough for perplexity - 1000+ words)
        default_text = """Artificial intelligence has transformed the way we interact with technology in profound and unprecedented ways. From natural language processing to computer vision, machine learning algorithms have enabled machines to understand and generate human-like content with remarkable accuracy. Large language models, trained on vast amounts of text data, can now perform a wide variety of tasks including text generation, translation, summarization, question answering, and even creative writing. These models learn patterns and relationships in data through sophisticated neural network architectures. The transformer architecture, introduced in the landmark paper "Attention is All You Need" in 2017, revolutionized the field by enabling models to process sequences of data in parallel rather than sequentially. This breakthrough led to the development of increasingly powerful models like GPT, BERT, T5, and their successors, each pushing the boundaries of what machines can accomplish.

The training process for these models involves exposing them to enormous datasets containing billions or even trillions of words from books, websites, scientific papers, and other text sources. During training, the model learns to predict the next word in a sequence, developing an understanding of grammar, facts, reasoning abilities, and even some level of common sense. This self-supervised learning approach allows models to learn from unlabeled data, which is far more abundant than labeled datasets. Fine-tuning these pre-trained models on specific tasks or domains can further improve their performance for particular applications. This approach, known as transfer learning, has become a standard practice in modern machine learning and has democratized access to powerful AI capabilities by allowing researchers and developers to build on existing models rather than training from scratch.

The scale of these models has grown exponentially over the years. Early language models had millions of parameters, while modern models can have hundreds of billions of parameters. This scaling has led to emergent capabilities that were not present in smaller models, such as few-shot learning, where models can perform new tasks with just a few examples, and chain-of-thought reasoning, where models can break down complex problems into steps. The relationship between model size, training data, and computational resources follows predictable scaling laws, which researchers use to guide the development of future models.

Despite their impressive capabilities, language models still face several significant challenges. They can sometimes generate incorrect or nonsensical information, a phenomenon known as hallucination. They struggle with tasks requiring deep reasoning, mathematical computations, or factual accuracy about recent events. Models may also reflect biases present in their training data, potentially amplifying societal biases related to gender, race, or other sensitive attributes. Researchers continue to work on improving model reliability, interpretability, and safety through various techniques. New approaches like reinforcement learning from human feedback help align model outputs with human preferences and values, making them more helpful, harmless, and honest.

The computational requirements for training these models are substantial. Training a large language model can cost millions of dollars in compute resources and consume vast amounts of energy. This has led to increased focus on efficiency, with researchers exploring techniques like mixed-precision training, gradient checkpointing, and model parallelism to reduce costs and environmental impact. Inference efficiency is also critical, as deployed models need to respond quickly while using minimal resources. Techniques like quantization, pruning, and knowledge distillation can compress models significantly while preserving most of their capabilities.

The applications of language models span numerous domains including customer service, content creation, code generation, education, research assistance, and creative arts. In customer service, chatbots powered by language models can handle complex queries and provide personalized responses. In content creation, they assist writers with brainstorming, editing, and generating drafts. For code generation, models can write, explain, and debug code across multiple programming languages. In education, they serve as tutors that can explain concepts, answer questions, and provide personalized learning experiences. Researchers use them to summarize papers, generate hypotheses, and even assist in writing manuscripts.

As these models continue to evolve, they're becoming more efficient, capable, and accessible to a broader range of users. The development of smaller, specialized models that excel at specific tasks provides alternatives to massive general-purpose models. Multimodal models that can process and generate both text and images represent an exciting frontier, enabling new applications in content creation, visual question answering, and embodied AI. The integration of retrieval mechanisms allows models to access external knowledge bases, addressing limitations of relying solely on parametric knowledge learned during training.

The future of AI likely involves even more sophisticated models that can understand context better, reason more effectively, and interact more naturally with humans across various mediums including text, speech, images, and video. Research directions include improving long-term memory and consistency, enabling better planning and multi-step reasoning, and developing models that can learn continuously from interactions rather than being static after training. The ethical implications of these powerful technologies require careful consideration, including issues of privacy, fairness, transparency, and the potential impact on employment and society at large. As the field advances, collaboration between researchers, policymakers, and the public will be essential to ensure that AI systems are developed and deployed responsibly for the benefit of all.

Machine learning has fundamentally changed how we approach problem-solving in computer science. Traditional programming requires explicitly defining rules and logic for every possible scenario, but machine learning allows systems to discover patterns and make decisions based on data. This paradigm shift has enabled breakthroughs in areas that were previously considered too complex for computational solutions. Image recognition, speech synthesis, natural language understanding, and game playing are just a few examples of domains where machine learning has achieved or exceeded human-level performance.

The evolution of deep learning has been particularly transformative. Deep neural networks, inspired by the structure of the human brain, consist of multiple layers of interconnected nodes that process information hierarchically. Each layer learns to recognize increasingly complex features, from simple edges and textures in early layers to sophisticated concepts in deeper layers. This hierarchical feature learning eliminates the need for manual feature engineering, which was a major bottleneck in traditional machine learning approaches.

Data quality and quantity are crucial factors in the success of machine learning models. High-quality, diverse, and representative training data leads to models that generalize well to new situations. However, collecting and curating such datasets is often expensive and time-consuming. Data augmentation techniques, synthetic data generation, and transfer learning help mitigate these challenges by making more efficient use of available data. Active learning approaches can identify the most informative examples to label, reducing the annotation burden while maintaining model performance.

The interpretability and explainability of machine learning models have become increasingly important as these systems are deployed in high-stakes applications. Black-box models, while often highly accurate, can be difficult to trust and debug when they make mistakes. Techniques for model interpretation, such as attention visualization, saliency maps, and feature importance analysis, help practitioners understand what factors influence model decisions. Explainable AI aims to make model behavior more transparent and understandable to both technical and non-technical users.

Ethical considerations in AI development extend beyond fairness and bias to include questions of accountability, privacy, and societal impact. Who is responsible when an AI system makes a harmful decision? How can we protect individual privacy while still enabling beneficial AI applications? What happens to jobs and industries disrupted by automation? These questions require interdisciplinary collaboration involving technologists, ethicists, policymakers, and affected communities. Developing frameworks for responsible AI that balance innovation with safety and equity is an ongoing challenge that will shape the future of the field."""
        
        batch_text = gr.Textbox(
            lines=15,
            max_lines=15,
            label="Evaluation Text (minimum 1024 tokens ≈ 750-800 words)",
            value=default_text.strip(),
            placeholder="Enter or paste a long text for perplexity evaluation...",
            show_label=True
        )
        batch_button = gr.Button("Run Batch Evaluation", variant="primary")
        batch_table = gr.Dataframe(headers=["Model", "Perplexity"], datatype=["str", "number"])
        batch_plot = gr.Plot()

        batch_button.click(
            batch_eval,
            inputs=batch_text,
            outputs=[batch_table, batch_plot],
        )

    with gr.Tab("Speed Benchmark"):
        gr.Markdown("### Measure Generation Speed")
        gr.Markdown("Test how fast each model generates tokens. Higher tokens/second = faster model!")
        
        benchmark_tokens = gr.Slider(
            50, 500, 
            value=200, 
            step=50, 
            label="Number of Tokens to Generate"
        )
        benchmark_button = gr.Button("Run Speed Test", variant="primary")
        
        speed_table = gr.Dataframe(
            headers=["Model", "Total Time", "Tokens/Second"],
            datatype=["str", "str", "number"]
        )
        speed_plot = gr.Plot()
        
        benchmark_button.click(
            speed_benchmark,
            inputs=benchmark_tokens,
            outputs=[speed_table, speed_plot],
        )

    with gr.Tab("Task Testing"):
        gr.Markdown("### Test Models on Specific Tasks")
        gr.Markdown("""
        Evaluate all models on standardized tasks:
        - **Math**: Basic calculations
        - **Reasoning**: Logical thinking
        - **Summarization**: Text compression
        - **Code**: Programming ability
        """)
        
        task_button = gr.Button("Run All Tasks", variant="primary")
        
        with gr.Row():
            with gr.Column():
                baseline_output = gr.Textbox(
                    lines=20,
                    label="Baseline (10K) Results",
                    show_label=True
                )
            with gr.Column():
                model_centric_output = gr.Textbox(
                    lines=20,
                    label="Model-Centric (10K) Results",
                    show_label=True
                )
            with gr.Column():
                data_centric_output = gr.Textbox(
                    lines=20,
                    label="Data-Centric (30K) Results",
                    show_label=True
                )
        
        task_button.click(
            few_shot_tasks,
            inputs=None,
            outputs=[baseline_output, model_centric_output, data_centric_output],
        )


demo.launch(server_name="0.0.0.0", server_port=7860)
