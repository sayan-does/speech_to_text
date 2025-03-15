import os
import sys
import torch
import whisper
from tqdm import tqdm
import time
import datetime
import wave
import contextlib
from pydub import AudioSegment
import argparse

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    timestamp = datetime.timedelta(seconds=seconds)
    return str(timestamp).replace(".", ",")[:11]

def get_audio_duration(audio_path):
    """Get duration of audio file using wave module"""
    try:
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error getting audio duration with wave: {e}")
        
        # Fallback to pydub if wave fails
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception as e2:
            print(f"Error getting audio duration with pydub: {e2}")
            raise

def extract_segment(audio_path, segment_start, segment_end, segment_file):
    """Extract a segment of audio using pydub"""
    try:
        audio = AudioSegment.from_file(audio_path)
        start_ms = int(segment_start * 1000)
        end_ms = int(segment_end * 1000)
        segment = audio[start_ms:end_ms]
        segment = segment.set_frame_rate(16000).set_channels(1)
        segment.export(segment_file, format="wav")
        return True
    except Exception as e:
        print(f"Error extracting segment: {e}")
        return False

def transcribe_audio(audio_path, model_name="medium", device=None, segment_length=30, output_path=None):
    """
    Transcribe audio using Whisper model and output SRT file
    
    Args:
        audio_path: Path to the input WAV file
        model_name: Whisper model size to use (tiny, base, small, medium, large)
        device: Device to use (cuda or cpu). If None, automatically selects
        segment_length: Length of audio segments to process in seconds
        output_path: Path for output SRT file. If None, uses same name as audio
        
    Returns:
        Path to the generated SRT file
    """
    # Set default device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Whisper model '{model_name}' on {device}...")
    model = whisper.load_model(model_name, device=device)
    
    print(f"Transcribing {audio_path}...")
    start_time = time.time()
    
    # Set default output path if not specified
    if output_path is None:
        output_path = os.path.splitext(audio_path)[0] + ".srt"
    
    # Get audio duration
    try:
        duration = get_audio_duration(audio_path)
        print(f"Audio duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        print("Using fallback method with full transcription...")
        
        # Fallback to full file transcription if we can't segment
        result = model.transcribe(audio_path)
        with open(output_path, 'w', encoding='utf-8') as srt_file:
            for i, segment in enumerate(result["segments"], 1):
                srt_file.write(f"{i}\n")
                srt_file.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
                srt_file.write(f"{segment['text'].strip()}\n\n")
        
        elapsed_time = time.time() - start_time
        print(f"Transcription completed in {elapsed_time:.2f} seconds (fallback method)")
        return output_path
    
    # Process audio in segments
    with open(output_path, 'w', encoding='utf-8') as srt_file:
        subtitle_index = 1
        
        # Process audio in segments with progress bar
        for segment_start in tqdm(range(0, int(duration), segment_length)):
            segment_end = min(segment_start + segment_length, duration)
            
            # Create temporary segment file
            segment_file = f"temp_segment_{segment_start}_{segment_end}.wav"
            
            try:
                # Extract segment
                if not extract_segment(audio_path, segment_start, segment_end, segment_file):
                    print(f"Skipping segment {segment_start}-{segment_end} due to extraction failure")
                    continue
                
                # Transcribe segment
                result = model.transcribe(segment_file, word_timestamps=True)
                
                # Write subtitles for this segment
                for segment in result["segments"]:
                    # Adjust timestamps to account for the segment start time
                    start_time_adjusted = segment_start + segment["start"]
                    end_time_adjusted = segment_start + segment["end"]
                    
                    # Format for SRT file
                    srt_file.write(f"{subtitle_index}\n")
                    srt_file.write(f"{format_timestamp(start_time_adjusted)} --> {format_timestamp(end_time_adjusted)}\n")
                    srt_file.write(f"{segment['text'].strip()}\n\n")
                    
                    subtitle_index += 1
                
            except Exception as e:
                print(f"Error processing segment {segment_start}-{segment_end}: {e}")
            finally:
                # Clean up temporary segment file
                if os.path.exists(segment_file):
                    os.remove(segment_file)
            
            # Show progress update
            print(f"Processed up to {segment_end:.1f}s / {duration:.1f}s ({(segment_end/duration*100):.1f}%)")
    
    elapsed_time = time.time() - start_time
    print(f"Transcription completed in {elapsed_time:.2f} seconds")
    print(f"SRT file saved to {output_path}")
    return output_path

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Transcribe audio to SRT subtitles using Whisper")
    parser.add_argument("audio_path", help="Path to the WAV audio file")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], 
                       default="medium", help="Whisper model size")
    parser.add_argument("--device", choices=["cuda", "cpu"], 
                       help="Device to use (default: auto-detect)")
    parser.add_argument("--segment", type=int, default=30,
                       help="Segment length in seconds (default: 30)")
    parser.add_argument("--output", "-o", help="Output SRT file path (optional)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Transcribe audio
    result = transcribe_audio(
        args.audio_path,
        model_name=args.model,
        device=args.device,
        segment_length=args.segment,
        output_path=args.output
    )
    
    # Check result
    if result:
        print("Transcription successful!")
        print(f"Subtitles saved to: {result}")
        return 0
    else:
        print("Transcription failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())