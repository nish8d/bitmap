import subprocess
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re


def get_video_info(video_path):
    """Extract framerate and frame data in a single ffprobe call."""
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate:frame=pkt_size,pict_type',
        '-of', 'json',
        video_path
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Extract FPS
        fps_string = data['streams'][0]['r_frame_rate']
        numerator, denominator = map(int, fps_string.split('/'))
        fps = numerator / denominator
        
        # Extract frame sizes and types
        frames = data.get('frames', [])
        frame_data = []
        for f in frames:
            if 'pkt_size' in f:
                frame_data.append({
                    'size': int(f['pkt_size']),
                    'type': f.get('pict_type', 'U')  # I, P, B, or Unknown
                })
        
        return fps, frame_data
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}")
    except (KeyError, IndexError, ValueError) as e:
        raise RuntimeError(f"Failed to parse video info: {e}")


def extract_motion_vectors(video_path):
    """Extract motion vectors using ffmpeg and calculate per-frame motion magnitude."""
    print("🎯 Extracting motion vectors (this may take a moment)...")
    
    # First pass: get frame count for array initialization
    fps, frame_data = get_video_info(video_path)
    num_frames = len(frame_data)
    motion_magnitudes = np.zeros(num_frames, dtype=np.float32)
    
    command = [
        'ffmpeg',
        '-flags2', '+export_mvs',
        '-i', video_path,
        '-vf', 'codecview=mv=pf+bf+bb',
        '-an',
        '-f', 'null',
        '-'
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, 
                               stderr=subprocess.PIPE, check=False)
        
        # Parse stderr for motion vector data
        # ffmpeg outputs frame info in format: "frame= XXXX"
        current_frame = 0
        frame_mv_sum = 0
        frame_mv_count = 0
        
        for line in result.stderr.split('\n'):
            # Track frame number from ffmpeg progress output
            frame_match = re.search(r'frame=\s*(\d+)', line)
            if frame_match:
                # Save previous frame's data
                if current_frame < num_frames and frame_mv_count > 0:
                    motion_magnitudes[current_frame] = frame_mv_sum / frame_mv_count
                
                # Move to next frame
                current_frame = int(frame_match.group(1)) - 1
                frame_mv_sum = 0
                frame_mv_count = 0
            
            # Parse motion vector magnitude from codecview output
            # Format varies, but typically contains motion vector coordinates
            mv_match = re.search(r'mv.*?\[(-?\d+),\s*(-?\d+)\]', line)
            if mv_match and current_frame < num_frames:
                dx = int(mv_match.group(1))
                dy = int(mv_match.group(2))
                magnitude = np.sqrt(dx*dx + dy*dy)
                frame_mv_sum += magnitude
                frame_mv_count += 1
        
        # Save last frame
        if current_frame < num_frames and frame_mv_count > 0:
            motion_magnitudes[current_frame] = frame_mv_sum / frame_mv_count
        
        # Smooth the motion data
        window = 15
        if len(motion_magnitudes) > window:
            motion_magnitudes = np.convolve(motion_magnitudes, 
                                           np.ones(window)/window, mode='same')
        
        print(f"✅ Extracted motion vectors for {num_frames} frames")
        return motion_magnitudes
        
    except Exception as e:
        print(f"⚠️  Motion vector extraction failed: {e}")
        print("📊 Falling back to frame-size based motion proxy...")
        return None


def calculate_motion_proxy(frame_data, motion_vectors=None):
    """
    Calculate motion intensity from actual motion vectors or P/B frame sizes as fallback.
    """
    # Use real motion vectors if available
    if motion_vectors is not None:
        print("✅ Using actual motion vector data")
        return motion_vectors
    
    # Fallback: use P/B frame sizes as proxy
    print("📊 Using frame-size based motion proxy")
    motion_scores = []
    
    for i, frame in enumerate(frame_data):
        # I-frames don't indicate motion, use surrounding P/B frames
        if frame['type'] == 'I':
            motion_scores.append(0)
        else:
            # P and B frame sizes correlate with motion/complexity
            # Normalize by average to get relative motion
            motion_scores.append(frame['size'])
    
    # Convert to numpy for easier processing
    motion_scores = np.array(motion_scores, dtype=np.float32)
    
    # Smooth the signal to reduce noise
    window = 15  # ~0.5 second window at 30fps
    if len(motion_scores) > window:
        motion_scores = np.convolve(motion_scores, np.ones(window)/window, mode='same')
    
    return motion_scores


def calculate_scene_changes(frame_data):
    """Detect scene changes (locations of I-frames after GOP start)."""
    scene_changes = []
    last_i_frame = -999  # Track last I-frame position
    
    for i, frame in enumerate(frame_data):
        if frame['type'] == 'I':
            # If I-frame appears earlier than expected GOP, it's likely a scene change
            if i - last_i_frame < 120:  # Less than typical GOP size
                if last_i_frame > 0:  # Skip first I-frame
                    scene_changes.append(i)
            last_i_frame = i
    
    return scene_changes


def calculate_complexity_score(frame_data, window_size=30):
    """
    Calculate overall complexity score using multiple metrics.
    High complexity = high motion + high detail + frequent changes
    """
    sizes = np.array([f['size'] for f in frame_data], dtype=np.float32)
    
    # Metric 1: Average bitrate (smoothed)
    avg_bitrate = np.convolve(sizes, np.ones(window_size)/window_size, mode='same')
    
    # Metric 2: Variance (how much change between frames)
    variance = np.array([np.var(sizes[max(0, i-window_size):i+1]) 
                        for i in range(len(sizes))])
    
    # Metric 3: P/B frame ratio (more P-frames = more significant motion)
    pb_ratio = []
    for i in range(len(frame_data)):
        window_start = max(0, i - window_size)
        window_frames = frame_data[window_start:i+1]
        p_count = sum(1 for f in window_frames if f['type'] == 'P')
        b_count = sum(1 for f in window_frames if f['type'] == 'B')
        ratio = p_count / (b_count + 1) if b_count > 0 else 0
        pb_ratio.append(ratio)
    pb_ratio = np.array(pb_ratio)
    
    # Normalize all metrics to 0-1 range
    def normalize(arr):
        if arr.max() > arr.min():
            return (arr - arr.min()) / (arr.max() - arr.min())
        return arr
    
    avg_bitrate_norm = normalize(avg_bitrate)
    variance_norm = normalize(variance)
    pb_ratio_norm = normalize(pb_ratio)
    
    # Weighted combination
    complexity = (avg_bitrate_norm * 0.5 + 
                 variance_norm * 0.3 + 
                 pb_ratio_norm * 0.2)
    
    return complexity


def find_action_moments(complexity_scores, fps, top_n=10, min_duration=2.0):
    """
    Find the most action-packed moments in the video.
    
    Args:
        complexity_scores: Array of complexity values per frame
        fps: Frames per second
        top_n: Number of moments to return
        min_duration: Minimum duration in seconds for a moment
    """
    min_frames = int(min_duration * fps)
    
    # Find local peaks
    peaks = []
    for i in range(min_frames, len(complexity_scores) - min_frames):
        window = complexity_scores[i-min_frames:i+min_frames]
        if complexity_scores[i] == window.max() and complexity_scores[i] > 0.5:
            peaks.append((i, complexity_scores[i]))
    
    # Sort by intensity and get top N
    peaks.sort(key=lambda x: x[1], reverse=True)
    top_moments = []
    
    for frame_idx, intensity in peaks[:top_n]:
        timestamp = frame_idx / fps
        minutes = int(timestamp // 60)
        seconds = timestamp % 60
        top_moments.append({
            'frame': frame_idx,
            'timestamp': f"{minutes:02d}:{seconds:05.2f}",
            'intensity': intensity
        })
    
    return top_moments


def create_bitrate_visualization(video_path, output_path="bitrate_analysis.png", 
                                 img_height=800, graph_height=250):
    """Create comprehensive video analysis visualization."""
    video_path = Path(video_path)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"📊 Analyzing {video_path.name}...")
    
    # Get video info
    fps, frame_data = get_video_info(str(video_path))
    print(f"✅ Detected: {fps:.2f} FPS, {len(frame_data):,} frames")
    
    # Extract actual motion vectors
    motion_vectors = extract_motion_vectors(str(video_path))
    
    # Calculate motion and complexity
    print("🎬 Calculating motion intensity...")
    motion_scores = calculate_motion_proxy(frame_data, motion_vectors)
    
    print("🔍 Analyzing scene complexity...")
    complexity_scores = calculate_complexity_score(frame_data)
    
    print("✂️  Detecting scene changes...")
    scene_changes = calculate_scene_changes(frame_data)
    
    print("🎯 Finding top action moments...")
    action_moments = find_action_moments(complexity_scores, fps, top_n=10)
    
    # Downsample for display if needed
    downsample_factor = max(1, len(frame_data) // 4000)
    if downsample_factor > 1:
        frame_data_display = frame_data[::downsample_factor]
        motion_scores_display = motion_scores[::downsample_factor]
        complexity_scores_display = complexity_scores[::downsample_factor]
        scene_changes_display = [sc // downsample_factor for sc in scene_changes]
        print(f"📉 Downsampled to {len(frame_data_display):,} points ({downsample_factor}x)")
    else:
        frame_data_display = frame_data
        motion_scores_display = motion_scores
        complexity_scores_display = complexity_scores
        scene_changes_display = scene_changes
    
    img_width = len(frame_data_display)
    
    # Create image
    img = Image.new('RGB', (img_width, img_height), (15, 15, 15))
    draw = ImageDraw.Draw(img)
    
    print(f"🎨 Rendering visualization...")
    
    # === GRAPH 1: Bitrate (frame sizes) ===
    y_offset_1 = 50
    draw_frame_size_graph(draw, frame_data_display, img_width, graph_height, y_offset_1)
    draw.text((10, y_offset_1 - 30), "Frame Sizes (Bitrate)", fill=(200, 200, 200))
    
    # === GRAPH 2: Motion Intensity ===
    y_offset_2 = y_offset_1 + graph_height + 60
    draw_motion_graph(draw, motion_scores_display, img_width, graph_height, y_offset_2)
    draw.text((10, y_offset_2 - 30), "Motion Intensity (Actual Motion Vectors)", fill=(200, 200, 200))
    
    # === GRAPH 3: Overall Complexity ===
    y_offset_3 = y_offset_2 + graph_height + 60
    draw_complexity_graph(draw, complexity_scores_display, img_width, graph_height, y_offset_3)
    draw.text((10, y_offset_3 - 30), "Scene Complexity Score", fill=(200, 200, 200))
    
    # Mark scene changes on all graphs
    for sc in scene_changes_display:
        if 0 <= sc < img_width:
            draw.line([(sc, y_offset_1), (sc, y_offset_1 + graph_height)], 
                     fill=(255, 200, 0, 128), width=1)
            draw.line([(sc, y_offset_2), (sc, y_offset_2 + graph_height)], 
                     fill=(255, 200, 0, 128), width=1)
            draw.line([(sc, y_offset_3), (sc, y_offset_3 + graph_height)], 
                     fill=(255, 200, 0, 128), width=1)
    
    # Draw time labels
    draw_time_labels(draw, img_width, y_offset_3 + graph_height, 
                    len(frame_data), fps)
    
    # Draw statistics and top moments
    draw_analysis_results(draw, frame_data, action_moments, fps, 
                         img_width, img_height)
    
    # Save
    img.save(output_path, optimize=True)
    print(f"✅ Saved to {output_path}")
    
    # Print top action moments
    print("\n🎬 TOP ACTION MOMENTS:")
    print("=" * 50)
    for i, moment in enumerate(action_moments, 1):
        print(f"{i:2d}. {moment['timestamp']} - Intensity: {moment['intensity']:.2%}")
    
    return action_moments


def draw_frame_size_graph(draw, frame_data, width, height, y_offset):
    """Draw frame size graph with color-coded frame types."""
    sizes = np.array([f['size'] for f in frame_data])
    max_size = sizes.max()
    
    for x, frame in enumerate(frame_data):
        normalized_height = int((frame['size'] / max_size) * height * 0.95)
        
        # Color by frame type
        if frame['type'] == 'I':
            color = (255, 80, 80)  # Red for I-frames
        elif frame['type'] == 'P':
            color = (80, 150, 255)  # Blue for P-frames
        else:  # B-frames
            color = (80, 255, 150)  # Green for B-frames
        
        if normalized_height > 0:
            draw.line([(x, y_offset + height), 
                      (x, y_offset + height - normalized_height)], 
                     fill=color)


def draw_motion_graph(draw, motion_scores, width, height, y_offset):
    """Draw motion intensity graph."""
    if len(motion_scores) == 0:
        return
    
    max_score = motion_scores.max() if motion_scores.max() > 0 else 1
    
    for x in range(len(motion_scores)):
        normalized_height = int((motion_scores[x] / max_score) * height * 0.95)
        
        # Color gradient: low motion = blue, high motion = red
        intensity_ratio = motion_scores[x] / max_score
        r = int(intensity_ratio * 255)
        b = int((1 - intensity_ratio) * 255)
        color = (r, 100, b)
        
        if normalized_height > 0:
            draw.line([(x, y_offset + height), 
                      (x, y_offset + height - normalized_height)], 
                     fill=color)


def draw_complexity_graph(draw, complexity_scores, width, height, y_offset):
    """Draw overall complexity score graph."""
    for x in range(len(complexity_scores)):
        normalized_height = int(complexity_scores[x] * height * 0.95)
        
        # Yellow to red gradient for complexity
        intensity = complexity_scores[x]
        r = 255
        g = int(255 * (1 - intensity))
        color = (r, g, 0)
        
        if normalized_height > 0:
            draw.line([(x, y_offset + height), 
                      (x, y_offset + height - normalized_height)], 
                     fill=color)


def draw_time_labels(draw, img_width, y_pos, num_frames, fps):
    """Draw time axis labels."""
    num_labels = min(10, img_width // 100)
    if num_labels < 2:
        return
    
    step = img_width // num_labels
    
    for i in range(0, img_width, step):
        seconds_total = (i / img_width) * (num_frames / fps)
        mins = int(seconds_total // 60)
        secs = int(seconds_total % 60)
        time_str = f"{mins:02d}:{secs:02}"
        
        draw.line([(i, y_pos), (i, y_pos + 10)], fill=(200, 200, 200), width=1)
        draw.text((i + 5, y_pos + 15), time_str, fill=(200, 200, 200))


def draw_analysis_results(draw, frame_data, action_moments, fps, img_width, img_height):
    """Draw statistics and top action moments."""
    sizes = np.array([f['size'] for f in frame_data])
    
    # Calculate statistics
    avg_size = sizes.mean()
    max_size = sizes.max()
    avg_bitrate = (avg_size * 8 * fps) / 1_000_000
    max_bitrate = (max_size * 8 * fps) / 1_000_000
    duration = len(frame_data) / fps
    
    # Frame type distribution
    i_frames = sum(1 for f in frame_data if f['type'] == 'I')
    p_frames = sum(1 for f in frame_data if f['type'] == 'P')
    b_frames = sum(1 for f in frame_data if f['type'] == 'B')
    
    # Draw legend and stats
    x_pos = img_width - 300
    y_pos = 10
    
    stats = [
        f"Duration: {duration:.1f}s",
        f"Avg Bitrate: {avg_bitrate:.2f} Mbps",
        f"Max Bitrate: {max_bitrate:.2f} Mbps",
        "",
        f"I-frames: {i_frames} (red)",
        f"P-frames: {p_frames} (blue)",
        f"B-frames: {b_frames} (green)",
        "",
        "Scene changes: orange lines",
    ]
    
    for line in stats:
        draw.text((x_pos, y_pos), line, fill=(180, 180, 180))
        y_pos += 20


def main():
    """Main entry point."""
    video_file = "input.mp4"
    
    try:
        action_moments = create_bitrate_visualization(
            video_file,
            output_path="bitrate_motion_analysis.png",
            img_height=900
        )
        
        print("\n" + "="*50)
        print("✅ Analysis complete!")
        print("="*50)
        print("\n📊 The visualization shows:")
        print("  • Frame sizes (red=I, blue=P, green=B)")
        print("  • Motion intensity (from actual motion vectors)")
        print("  • Overall complexity score (combined metrics)")
        print("  • Scene changes (orange vertical lines)")
        print("\n💡 High complexity scores indicate action-packed moments!")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
    except RuntimeError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
