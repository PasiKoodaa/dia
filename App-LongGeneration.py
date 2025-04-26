import gradio as gr
import os
from pathlib import Path
import warnings
import time
import re
import torch
import numpy as np
import soundfile as sf
from dia.model import Dia
from rich import print as printr
from rich.console import Console

console = Console()

def chunk_text(text, audio_prompt_text):
    """Split text into speaker-aware chunks based on [S1] and [S2] tags.
    Args:
        text (str): The input text to chunk
        audio_prompt_text (str): The audio prompt text to prepend to each chunk
        
    Returns:
        list: List of tuples (text chunk, silence flag)
    """
    # Clean up input text
    lines = text.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    chunks = []
    current_chunk = []
    pattern = r'\[(S[12])\]'
    
    # Process each line
    for i, line in enumerate(lines):
        # Check if line has a speaker tag
        match = re.search(pattern, line)
        if not match:
            # If no speaker tag, add to previous chunk or skip
            if current_chunk:
                current_chunk.append(line)
            continue
            
        speaker = match.group(1)
        current_chunk.append(line)
        
        # If we have an S1-S2 pair or reached the end
        if len(current_chunk) >= 2 or i == len(lines) - 1:
            # Create a chunk with the current dialogue
            chunk_text = "\n".join(current_chunk)
            # Add audio prompt text if provided
            if audio_prompt_text:
                chunk_text = audio_prompt_text + "\n" + chunk_text
            
            # End with the next speaker tag to prevent audio shortening
            if not chunk_text.strip().endswith("[S1]") and not chunk_text.strip().endswith("[S2]"):
                next_speaker = "[S1]" if speaker == "S2" else "[S2]"
                chunk_text = chunk_text + "\n" + next_speaker
                
            # Check for ellipsis to determine silence flag
            silence_flag = not chunk_text.endswith("...")
            
            chunks.append((chunk_text, silence_flag))
            # Reset for next pair but keep current speaker tag if we have odd number
            if len(current_chunk) % 2 != 0:
                current_chunk = [current_chunk[-1]]
            else:
                current_chunk = []
    
    # Handle any remaining dialogue
    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        if audio_prompt_text:
            chunk_text = audio_prompt_text + "\n" + chunk_text
        chunks.append((chunk_text, True))
    
    return chunks

def add_silence(audio, duration_sec=0.5, sample_rate=44100):
    """Add silence to the end of an audio segment"""
    silence_samples = int(duration_sec * sample_rate)
    silence = np.zeros(silence_samples, dtype=audio.dtype)
    return np.concatenate([audio, silence])

def detect_device():
    """Detect the best available device for inference"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def get_duration(start):
    """Helper to calculate duration of the audio generation"""
    end = time.time()
    elapsed = end - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return minutes, seconds

def generate_with_retry(model, chunk, audio_prompt, args, max_retries=2):
    """
    Generate the audio of a dialogue chunk with retry logic for clamping warnings
    """
    retries = 0
    while retries <= max_retries:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Capture all warnings
            with torch.inference_mode():
                audio = model.generate(
                    text=chunk,
                    max_tokens=args.tokens_per_chunk,
                    cfg_scale=args.cfg_scale,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    cfg_filter_top_k=args.cfg_filter_top_k,
                    use_torch_compile=True,
                    audio_prompt=audio_prompt
                )

            # Check for the specific warning
            clamping_warning = any(
                "Clamping" in str(warning.message)
                for warning in w
            )

            if clamping_warning:
                printr("[red]⚠️ Clamping warning caught. Retrying generation...[/]")
                retries += 1
                continue  # Retry the loop
            else:
                break  # Success, exit loop

    if retries > max_retries:
        printr("[red]⚠️ Max retries reached. Returning last generated audio.[/]")
        
    return audio

class Args:
    """Simple class to hold arguments for the model."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def generate_audio(
    dialogue_text, 
    model_name="nari-labs/Dia-1.6B",
    speed=0.9, 
    silence=0.3, 
    tokens_per_chunk=3072, 
    cfg_scale=3.0, 
    temperature=1.3, 
    top_p=0.95, 
    cfg_filter_top_k=30,
    audio_prompt=None,
    text_prompt="",
    progress=gr.Progress()
):
    """Main function to generate audio from text for the Gradio interface."""
    output = []
    
    # Setup device
    device = detect_device()
    output.append(f"Using device: {device}")
    
    # Create a directory to save chunks
    app_dir = Path(__file__).parent
    chunks_dir = app_dir / "audio_chunks"
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Check if audio prompt and text prompt are both provided
    if audio_prompt is not None and audio_prompt != "" and text_prompt != "":
        audio_prompt_text = text_prompt
    else:
        if audio_prompt is not None and audio_prompt != "" and text_prompt == "":
            output.append("Warning: Text prompt is required when using an audio prompt. Voice cloning disabled.")
        text_prompt = ""
        audio_prompt_text = ""
        audio_prompt = None
    
    # Split text into chunks
    chunks = chunk_text(dialogue_text, audio_prompt_text)
    output.append(f"Split text into {len(chunks)} chunks")
    
    # Load model
    output.append(f"Loading Dia model from {model_name}...")
    progress(0.1, desc="Loading model")
    start_time = time.time()
    try:
        model = Dia.from_pretrained(model_name, compute_dtype="float16", device=device)
        output.append(f"Model loaded in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        output.append(f"Error loading model: {e}")
        return "\n".join(output), None
    
    # Set up arguments
    args = Args(
        tokens_per_chunk=tokens_per_chunk,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_p=top_p,
        cfg_filter_top_k=cfg_filter_top_k,
        speed=speed,
        silence=silence
    )
    
    # Generate audio for each chunk
    tmp_files = []
    output.append(f"Generating audio for each chunk...")
    total_start = time.time()
    
    for i, (chunk, silence_flag) in enumerate(chunks):
        chunk_file = chunks_dir / f"chunk_{i:03d}.wav"
        tmp_files.append(chunk_file)
        
        # Update progress
        progress((i / len(chunks)) * 0.8 + 0.1, desc=f"Processing chunk {i+1}/{len(chunks)}")
        
        # Print chunk info
        output.append(f"\nChunk {i+1}/{len(chunks)}")
        
        if not silence_flag:
            output.append("Silence removed due to [...] detected at the end of the chunk")
        
        # Show a preview of the chunk
        chunk_preview = chunk[:200]
        if len(chunk) > 200:
            chunk_preview += " [...]"
        output.append(f"{'='*40}\n{chunk_preview}\n{'='*40}")
        
        # Generate audio for this chunk
        start_time = time.time()
        try:
            # Audio generation with retry logic
            audio = generate_with_retry(model, chunk, audio_prompt, args)
            
            # Apply speed adjustment
            if speed != 1.0:
                orig_len = len(audio)
                target_len = int(orig_len / speed)
                x_orig = np.arange(orig_len)
                x_new = np.linspace(0, orig_len-1, target_len)
                audio = np.interp(x_new, x_orig, audio)
            
            # Add silence at the end of the audio fragment
            if silence > 0 and silence_flag:
                audio = add_silence(audio, silence)
            
            # Save chunk file
            sf.write(chunk_file, audio, 44100)
            
            # Generation statistics
            minutes, seconds = get_duration(start_time)
            if minutes > 0:
                output.append(f"Generated chunk {i+1} (duration: {len(audio)/44100:.2f} seconds) - Processed in {minutes} minutes and {seconds} seconds")
            else:
                output.append(f"Generated chunk {i+1} (duration: {len(audio)/44100:.2f} seconds) - Processed in {seconds} seconds")
        
        except Exception as e:
            output.append(f"Error processing chunk {i+1}: {e}")
    
    # Combine all audio files
    progress(0.9, desc="Combining audio segments")
    output.append(f"Combining {len(tmp_files)} audio segments...")
    all_audio = []
    
    for tmp_file in tmp_files:
        if tmp_file.exists():
            audio, sr = sf.read(tmp_file)
            all_audio.append(audio)
    
    if not all_audio:
        output.append("Error: No audio was generated")
        return "\n".join(output), None
    
    # Concatenate and save the final output
    final_audio = np.concatenate(all_audio)
    
    # Create final output file
    output_dir = app_dir / "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / "final_output.wav"
    output.append(f"Saving final audio (duration: {len(final_audio)/44100:.2f} seconds)")
    sf.write(output_file, final_audio, 44100)
    
    minutes, seconds = get_duration(total_start)
    output.append(f"Done! Total processing time: {minutes} minutes and {seconds} seconds")
    
    progress(1.0, desc="Processing complete")
    return "\n".join(output), str(output_file)

# Create the Gradio interface
with gr.Blocks(title="Dia TTS - Dialogue Text to Speech") as app:
    gr.Markdown("# Dia TTS - Convert Dialogue to Natural Speech")
    gr.Markdown("""
    This app uses the Dia text-to-speech model to convert dialogue text into natural-sounding speech.
    
    ## How to format your dialogue:
    - Use `[S1]` and `[S2]` to mark different speakers
    - Each speaker should be on a separate line
    - Example:
    ```
    [S1] Hello, how are you today?
    [S2] I'm doing well, thank you for asking. How about yourself?
    [S1] I'm great, thanks! I was wondering if you had time to discuss the project.
    [S2] Of course, I'd be happy to talk about it.
    ```
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Model Settings")
                model_name = gr.Dropdown(
                    choices=["nari-labs/Dia-1.6B"], 
                    value="nari-labs/Dia-1.6B", 
                    label="Model"
                )
                speed = gr.Slider(
                    minimum=0.7, 
                    maximum=1.3, 
                    value=0.95, 
                    step=0.05, 
                    label="Speech Speed (lower is slower)"
                )
                silence = gr.Slider(
                    minimum=0, 
                    maximum=1.0, 
                    value=0.3, 
                    step=0.05, 
                    label="Silence Between Chunks (seconds)"
                )
            
            with gr.Group():
                gr.Markdown("### Advanced Model Parameters")
                tokens_per_chunk = gr.Slider(
                    minimum=1024, 
                    maximum=4096, 
                    value=3072, 
                    step=256, 
                    label="Max Tokens Per Chunk"
                )
                cfg_scale = gr.Slider(
                    minimum=1.0, 
                    maximum=5.0, 
                    value=3.0, 
                    step=0.1, 
                    label="CFG Scale"
                )
                temperature = gr.Slider(
                    minimum=0.5, 
                    maximum=2.0, 
                    value=1.3, 
                    step=0.1, 
                    label="Temperature"
                )
                top_p = gr.Slider(
                    minimum=0.5, 
                    maximum=1.0, 
                    value=0.95, 
                    step=0.05, 
                    label="Top P"
                )
                cfg_filter_top_k = gr.Slider(
                    minimum=5, 
                    maximum=50, 
                    value=30, 
                    step=5, 
                    label="CFG Filter Top K"
                )
            
            with gr.Group():
                gr.Markdown("### Voice Cloning (Optional)")
                with gr.Row():
                    audio_prompt = gr.Audio(
                        type="filepath", 
                        label="Audio Prompt for Voice Cloning"
                    )
                text_prompt = gr.Textbox(
                    placeholder="Enter text that matches the audio prompt...", 
                    label="Text Prompt for Voice Cloning"
                )
            
        with gr.Column(scale=1):
            dialogue_text = gr.Textbox(
                placeholder="Enter dialogue text here...", 
                label="Dialogue Text",
                lines=15
            )
            generate_button = gr.Button("Generate Audio", variant="primary")
            
            with gr.Group():
                output_log = gr.Textbox(label="Processing Log", lines=10)
                output_audio = gr.Audio(label="Generated Audio")
    
    # Set up the generation function
    inputs = [
        dialogue_text, 
        model_name,
        speed, 
        silence, 
        tokens_per_chunk, 
        cfg_scale, 
        temperature, 
        top_p, 
        cfg_filter_top_k,
        audio_prompt,
        text_prompt
    ]
    
    outputs = [output_log, output_audio]
    
    generate_button.click(
        fn=generate_audio, 
        inputs=inputs, 
        outputs=outputs
    )
    
    # Example
    example_dialogue = """[S1] Hello, how are you today?
[S2] I'm doing well, thank you for asking. How about yourself?
[S1] I'm great, thanks! I was wondering if you had time to discuss the project.
[S2] Of course, I'd be happy to talk about it. What aspects would you like to focus on?
[S1] I'm particularly interested in the timeline and key milestones. Do you think we're on track?
[S2] Based on our current progress, I believe we're slightly ahead of schedule. We've completed the initial research phase faster than anticipated.
[S1] That's excellent news! What about the budget considerations? Are we still within our allocated resources?
[S2] Yes, we're currently under budget by approximately 5%. However, we should keep in mind that the next phase might require additional investments in specialized equipment."""

    gr.Examples(
        [[example_dialogue]],
        [dialogue_text]
    )
    
    gr.Markdown("""
    ## Notes
    - The generation process may take some time depending on your hardware.
    - For best results, use a machine with GPU support.
    - Voice cloning requires both an audio file and matching text.
    """)

if __name__ == "__main__":
    app.launch()
