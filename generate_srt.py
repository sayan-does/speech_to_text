import os
import sys
import argparse
from audio_extractor import extract_audio
from whisper_transcriber import transcribe_audio

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate SRT subtitles from video files using Whisper")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], 
                       default="medium", help="Whisper model size")
    parser.add_argument("--segment", type=int, default=30,
                       help="Segment length in seconds (default: 30)")
    parser.add_argument("--keep-audio", action="store_true",
                       help="Keep the extracted audio file")
    parser.add_argument("--output-dir", help="Output directory (default: same as video)")
    parser.add_argument("--audio-only", action="store_true",
                       help="Only extract audio, don't transcribe")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up output paths
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.video_path))[0]
        audio_path = os.path.join(args.output_dir, f"{base_name}.wav")
        srt_path = os.path.join(args.output_dir, f"{base_name}.srt")
    else:
        video_dir = os.path.dirname(args.video_path)
        base_name = os.path.splitext(os.path.basename(args.video_path))[0]
        audio_path = os.path.join(video_dir, f"{base_name}.wav")
        srt_path = os.path.join(video_dir, f"{base_name}.srt")
    
    # Extract audio
    print("\n=== AUDIO EXTRACTION ===")
    audio_result = extract_audio(args.video_path, audio_path)
    
    if not audio_result:
        print("Audio extraction failed. Cannot continue.")
        return 1
    
    # Stop here if only extracting audio
    if args.audio_only:
        print("\nAudio extraction completed successfully.")
        print(f"Audio saved to: {audio_result}")
        return 0
    
    # Transcribe audio
    print("\n=== AUDIO TRANSCRIPTION ===")
    transcribe_result = transcribe_audio(
        audio_result,
        model_name=args.model,
        segment_length=args.segment,
        output_path=srt_path
    )
    
    # Clean up audio file if not keeping it
    if not args.keep_audio and os.path.exists(audio_result):
        print("\nCleaning up temporary files...")
        os.remove(audio_result)
        print(f"Removed temporary audio file: {audio_result}")
    
    # Check final result
    if transcribe_result:
        print("\nSubtitle generation completed successfully!")
        print(f"Subtitles saved to: {transcribe_result}")
        return 0
    else:
        print("\nSubtitle generation failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())