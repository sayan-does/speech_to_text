import os
import sys
from moviepy import VideoFileClip
import argparse

def extract_audio(video_path, output_path=None):
    """
    Extract audio from video file using MoviePy
    
    Args:
        video_path: Path to input video file
        output_path: Path for output WAV file. If None, uses the same name as video
        
    Returns:
        Path to the extracted audio file
    """
    try:
        if output_path is None:
            output_dir = os.path.dirname(video_path)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.wav")
        
        #output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        print(f"Extracting audio from {video_path}...")
        print(f"Output will be saved to {output_path}")
        
        # Load video and extract audio
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(
            output_path,
            fps=16000,           # Sample rate: 16kHz
            nbytes=2,            # 16-bit audio
            codec='pcm_s16le',   # PCM format
            ffmpeg_params=['-ac', '1']  # Mono channel
        )
        video.close()
        
        print(f"Audio extraction completed: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error extracting audio: {e}")
        
        # Try alternate method using MoviePy's simpler approach
        try:
            print("Trying alternate extraction method...")
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(output_path)
            video.close()
            print(f"Audio extracted with alternate method: {output_path}")
            return output_path
        except Exception as alt_e:
            print(f"Alternative extraction also failed: {alt_e}")
            print("Please make sure MoviePy and its dependencies are properly installed.")
            return None

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Extract audio from video files")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output", "-o", help="Output WAV file path (optional)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract audio
    result = extract_audio(args.video_path, args.output)
    
    # Check result
    if result:
        print("Audio extraction successful!")
        print(f"Audio saved to: {result}")
        return 0
    else:
        print("Audio extraction failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())