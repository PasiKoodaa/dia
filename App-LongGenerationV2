import gradio as gr
import os
from pathlib import Path
import warnings
import time
import re
import torch
import numpy as np
import soundfile as sf
import whisperx
import gc
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dia.model import Dia
from rich import print as printr
from rich.console import Console

console = Console()

def is_url(text):
    """Check if the provided text is a URL"""
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def extract_text_from_url(url):
    """Fetch text content from a URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Extract text
        text = soup.get_text(separator='\n')
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

def send_to_koboldcpp(text):
    """Send text to KoboldCPP server and get generated dialogue"""
    url = "http://localhost:5001/api/v1/generate"
    
    system_prompt = "\nEND\n\nYour mission: Create a dialogue between two people in a podcast format based on the previous article. [S1], the host, and [S2], the guest speaker is an expert in the field but not the author of the paper, should engage in a conversation to discuss and simplify the paper's content for the listeners. Allowed verbal tags in created dialogue are: (laughs), (sighs), (gasps), (coughs), (groans), (sniffs), (inhales), (exhales), (whistles). Exclamation points are not allowed. Speaker shouldn't reference to another spearker with tag [S1] or [S2]. Remember to create the dialogue in form [S1] dialogue [S2] dialogue"
    
    full_text = text + system_prompt
    
    data = {
        "prompt": full_text,
        "max_context_length": 16000,
        "max_length": 1000,
        "temperature": 0.3
    }
    try:
        response = requests.post(url, json=data)
        return response.json()["results"][0]["text"]
    except requests.RequestException as e:
        return f"Error: {str(e)}"

def transcribe_audio_with_whisperx(audio_file, hf_token=None):
    """
    Transcribe audio using WhisperX with speaker diarization
    
    Args:
        audio_file (str): Path to the audio file
        hf_token (str, optional): HuggingFace token for diarization model
        
    Returns:
        str: Transcribed text with speaker tags [S1], [S2], etc.
    """

    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int32"
        batch_size = 16 if torch.cuda.is_available() else 4
        
        # 1. Transcribe with Whisper
        printr(f"[blue]Loading WhisperX model on {device}...[/]")
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        
        printr(f"[blue]Transcribing audio file: {audio_file}[/]")
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=batch_size)
        
        # Free GPU memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. Align whisper output
        printr(f"[blue]Aligning transcription in language: {result['language']}[/]")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # Free GPU memory
        del model_a
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 3. Assign speaker labels
        printr("[blue]Performing speaker diarization...[/]")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Format transcription with [S1], [S2], etc.
        formatted_text = []
        prev_speaker = None
        
        for segment in result["segments"]:
            speaker = segment.get("speaker", "SPEAKER_UNKNOWN")
            
            # Convert speaker to S1, S2 format
            if speaker == "SPEAKER_00" or speaker == "SPEAKER_0":
                speaker_tag = "[S1]"
            elif speaker == "SPEAKER_01" or speaker == "SPEAKER_1":
                speaker_tag = "[S2]"
            else:
                # Handle more than 2 speakers (though Dia only supports 2)
                speaker_num = int(speaker.split("_")[-1]) + 1
                speaker_tag = f"[S{speaker_num}]"
            
            # Only add speaker tag when speaker changes
            if speaker != prev_speaker:
                formatted_text.append(f"{speaker_tag} {segment['text']}")
            else:
                formatted_text.append(segment['text'])
            
            prev_speaker = speaker
        
        return "\n".join(formatted_text)
        
    except Exception as e:
        printr(f"[red]Error in WhisperX processing: {str(e)}[/]")
        return f"Transcription failed: {str(e)}"

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

def on_audio_upload(audio_file, hf_token, text_prompt, progress=gr.Progress()):
    """Handle audio uploads by automatically transcribing them only if text field is empty"""
    if audio_file is None:
        return text_prompt  # Return unchanged if no audio file
    
    # Only transcribe if the text prompt is empty
    if text_prompt.strip() == "":
        progress(0.1, desc="Loading WhisperX...")
        # Auto-transcribe the uploaded audio
        progress(0.3, desc="Transcribing audio...")
        transcription = transcribe_audio_with_whisperx(audio_file, hf_token)
        progress(1.0, desc="Transcription complete")
        return transcription
    else:
        # If text prompt is not empty, respect user's input
        return text_prompt

def process_input_text(input_text, progress=gr.Progress()):
    """Process input text (URL or direct text)"""
    if not input_text:
        return "No input provided"
    
    progress(0.1, desc="Processing input...")
    
    # Check if the input is a URL
    if is_url(input_text):
        progress(0.3, desc="Fetching content from URL...")
        content = extract_text_from_url(input_text)
    else:
        # Use the input directly as content
        content = input_text
    
    progress(0.7, desc="Sending to KoboldCPP...")
    # Generate dialogue using KoboldCPP
    dialogue = send_to_koboldcpp(content)
    
    progress(1.0, desc="Processing complete")
    return dialogue

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
    hf_token="",
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
    
    # Check if audio prompt is provided but text prompt is missing
    if audio_prompt is not None and audio_prompt != "" and text_prompt == "":
        output.append("Auto-transcribing audio prompt using WhisperX...")
        progress(0.05, desc="Auto-transcribing audio prompt")
        try:
            text_prompt = transcribe_audio_with_whisperx(audio_prompt, hf_token)
            output.append("Transcription successful!")
            output.append(f"Transcribed text: {text_prompt[:200]}...")
        except Exception as e:
            output.append(f"Error in auto-transcription: {str(e)}")
            output.append("Warning: Text prompt is required when using an audio prompt. Voice cloning disabled.")
            audio_prompt = None
    
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
    
    with gr.Tabs():
        with gr.TabItem("Generate Speech"):
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
                        with gr.Row():
                            text_prompt = gr.Textbox(
                                placeholder="Enter text that matches the audio prompt... (Will be auto-generated if empty)", 
                                label="Text Prompt for Voice Cloning",
                                lines=5
                            )
                        
                        hf_token = gr.Textbox(
                            placeholder="Enter your HuggingFace token for diarization (required for speaker identification)", 
                            label="HuggingFace Token",
                            type="password"
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
        
        with gr.TabItem("Generate Dialogue"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Source")
                    gr.Markdown("""
                    Enter either:
                    1. A URL to fetch content (e.g., scientific paper, article)
                    2. Direct text content
                    
                    The content will be processed by KoboldCPP to generate a dialogue in podcast format.
                    """)
                    
                    input_text = gr.Textbox(
                        placeholder="Enter URL or paste text content here...", 
                        label="Input (URL or Text)",
                        lines=15
                    )
                    process_button = gr.Button("Generate Dialogue", variant="primary")
                
                with gr.Column(scale=1):
                    generated_dialogue = gr.Textbox(
                        placeholder="Generated dialogue will appear here...", 
                        label="Generated Dialogue",
                        lines=15
                    )
                    copy_to_tts_button = gr.Button("Copy to Text-to-Speech")

    # Connect components
    
    # Auto-transcribe when audio is uploaded (only if text field is empty)
    audio_prompt.change(
        fn=on_audio_upload,
        inputs=[audio_prompt, hf_token, text_prompt],  # Added text_prompt to inputs
        outputs=text_prompt
    )
    
    # Process input text or URL
    process_button.click(
        fn=process_input_text,
        inputs=[input_text],
        outputs=generated_dialogue
    )
    
    # Copy generated dialogue to TTS input
    copy_to_tts_button.click(
        fn=lambda x: x,
        inputs=[generated_dialogue],
        outputs=[dialogue_text]
    )
    
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
        text_prompt,
        hf_token
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
    
    example_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    gr.Examples(
        [[example_url]],
        [input_text]
    )
    
    gr.Markdown("""
    ## Notes
    - The generation process may take some time depending on your hardware.
    - For best results, use a machine with GPU support.
    - **Voice cloning with automatic transcription**: Upload an audio file, and the app will use WhisperX to identify speakers and transcribe it automatically.
    - WhisperX requires significant processing power - transcription will be much faster with a GPU.
    - First-time WhisperX usage will download models which may take several minutes.
    - Make sure KoboldCPP server is running on http://localhost:5001 for the dialogue generation feature.
    """)

if __name__ == "__main__":
    app.launch()
