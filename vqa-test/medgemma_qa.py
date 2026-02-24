"""
MedGemma 1.5 Evaluation Script for Echocardiogram Video Dataset
 
This script runs MedGemma on a dataset of echocardiogram videos with multiple choice questions.
Since MedGemma is designed for images, we extract representative frames from videos.
 
Requirements:
    pip install accelerate transformers pillow requests torch opencv-python tqdm --break-system-packages
"""
 
import json
import os
import torch
import cv2
import argparse
import re
import math
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional
from transformers import AutoProcessor, AutoModelForImageTextToText
 
 
def load_model(model_id: str):#"google/medgemma-1.5-4b-it"):
    """Load MedGemma model and processor."""
    print(f"Loading model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model loaded successfully!")
    return model, processor
 
 
def extract_frames_from_video(
    video_path: str,
    num_frames: int = 1,
    strategy: str = "first"
) -> list[Image.Image]:
    """
    Extract frames from a video file.
   
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        strategy: How to select frames - 'middle', 'uniform', or 'first'
   
    Returns:
        List of PIL Images
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
   
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")
   
    # Determine which frame indices to extract
    if strategy == "middle":
        indices = [total_frames // 2]
    elif strategy == "first":
        indices = [0]
    elif strategy == "uniform":
        if num_frames >= total_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            indices = [int(i * step) for i in range(num_frames)]
    else:
        indices = [total_frames // 2]  # Default to middle
   
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
   
    cap.release()
    return frames
 
def extract_and_concat_frames(
    video_paths: list[str],
    strategy: str = "first",
) -> list[Image.Image]:
    """
    Extract one frame from each video and concatenate into a 2x2 grid.
    
    Args:
        video_paths: List of paths to video files
        strategy: Frame selection strategy - 'first' or 'middle'
        tile_size: Size to resize each frame before tiling
    
    Returns:
        List containing a single concatenated PIL Image
    """
    frames = []
    for path in video_paths:
        frame_list = extract_frames_from_video(path, num_frames=1, strategy=strategy)
        if frame_list:
            frames.append(frame_list[0])
    
    if not frames:
        raise ValueError("No frames could be extracted from any video")
    
    n = len(frames)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    # Use max width/height across frames to size each cell
    max_w = max(f.width for f in frames)
    max_h = max(f.height for f in frames)
    
    grid = Image.new("RGB", (max_w * cols, max_h * rows))
    for i, img in enumerate(frames):
        r, c = divmod(i, cols)
        grid.paste(img, (c * max_w, r * max_h))
    
    return [grid]

def format_multiple_choice_prompt(sample: dict, include_report: bool = False, multi_image: bool = False) -> str:
    """
    Format the question with multiple choice options.
   
    Args:
        sample: Dataset sample containing question and options
        include_report: Whether to include the clinical report as context
    """
    prompt_parts = []
   
    # Optionally include clinical context
    if include_report and sample.get("generated_report"):
        prompt_parts.append(f"Clinical Report:\n{sample['generated_report']}\n")
    # Add the question
    prompt_parts.append(f"Question: {sample['question']}")
   
    # Add the options
    prompt_parts.append("\nOptions:")
    for opt in ["A", "B", "C", "D"]:
        option_key = f"option_{opt}"
        if option_key in sample:
            prompt_parts.append(f"  {opt}. {sample[option_key]}")

    # Add instruction for the model
    if not multi_image:
        if include_report and sample.get("generated_report"):
            prompt_parts.append("\nAnswer with only the letter (A, B, C, or D) by looking at the report and image provided.")
            print('including report and single image...')
        else:
            prompt_parts.append("\nAnswer with only the letter (A, B, C, or D).")
            print('not including report and single image...')
    else:
        if include_report and sample.get("generated_report"):
            prompt_parts.append("\nAnswer with only the letter (A, B, C, or D) by looking at the report and 4 images provided.")
            print('including report and multi-image...')
        else:
            prompt_parts.append("\nAnswer with only the letter (A, B, C, or D).")
            print('not including report and multi-image...')

    return "\n".join(prompt_parts)
 
 
def run_inference(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 2000
) -> str:
    """Run inference with MedGemma on a single image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
   
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
   
    input_len = inputs["input_ids"].shape[-1]
   
    with torch.inference_mode():
        # print(f"Input shape: {inputs['input_ids'].shape}")
        # print(f"Device: {inputs['input_ids'].device}")
        import time
        # start = time.time()
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        generation = generation[0][input_len:]#.squeeze()
        # print(f"Generation took: {time.time() - start:.2f}s")
    # print(f"Generation type: {type(generation)}, shape: {generation.shape if hasattr(generation, 'shape') else 'N/A'}")
    decoded = processor.decode(generation, skip_special_tokens=True)
    return decoded
 
 
def extract_answer_letter(s: str) -> Optional[str]:

    # Priority 0: Letter between curly braces {A}
    match = re.search(r'\{([A-Za-z])\}', s)
    if match:
        return match.group(1).upper()

    # Priority 1: "Final answer: A"
    match = re.search(r'[Ff]inal [Aa]nswer:\s*([A-Za-z])\b', s)
    if match:
        return match.group(1).upper()

    # Priority 1.5:
    match = re.search(r"(?:therefore,?\s*)?the\s+(?:best|most appropriate|most accurate|correct)\s+answer is[:\s]*\n*\s*([A-Z])", s)
    if match:
        return match.group(1).upper()

    # Priority 2: **A (bold markdown)
    match = re.search(r'\*\*([A-Za-z])\b', s)
    if match:
        return match.group(1).upper()
    
    # Priority 2.5: **(A) (bold with parentheses)
    match = re.search(r'\*\*\(([A-Za-z])\)', s)
    if match:       
        return match.group(1).upper()

    # Priority 3: \nA (letter on its own line)
    match = re.search(r'(?:^|\n)\s*([A-Za-z])\s*(?:\n|$)', s)
    if match:
        return match.group(1).upper()

    # Priority 4: "The answer is A"
    match = re.search(r'[Tt]he answer is\s*([A-Za-z])\b', s)
    if match:
        return match.group(1).upper()

    return None
 
 
def evaluate_dataset(
    dataset_path: str,
    video_base_path: str,
    model,
    processor,
    output_path: str,
    include_report: bool = False,
    num_frames: int = 1,
    frame_strategy: str = "middle",
    max_samples: Optional[int] = None,
    multi_image: bool = False,
):
    """
    Evaluate MedGemma on the entire dataset.
   
    Args:
        dataset_path: Path to the JSON dataset file
        video_base_path: Base path where video files are stored
        model: Loaded MedGemma model
        processor: Loaded processor
        output_path: Path to save results
        include_report: Whether to include clinical reports in prompts
        num_frames: Number of frames to extract per video
        frame_strategy: Frame extraction strategy
        max_samples: Maximum number of samples to process (None for all)
        multi_image: Whether to input multiple images to the model
    """
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
   
    if isinstance(dataset, dict):
        # If dataset is a dict with a key containing the samples
        for key in ["data", "samples", "test", "items"]:
            if key in dataset:
                dataset = dataset[key]
                break
   
    if max_samples:
        dataset = dataset[:max_samples]
   
    results = []
    correct = 0
    total = 0
    skipped = 0
   
    for sample in tqdm(dataset, desc="Processing samples"):
        result = {
            "messages_id": sample.get("messages_id"),
            "question": sample.get("question"),
            "correct_option": sample.get("correct_option"),
            "answer": sample.get("answer"),
        }
       
        try:
            # Get video path
            videos = sample.get("videos", [])
            if not videos:
                print(f"No video found for sample {sample.get('messages_id')}")
                result["status"] = "no_video"
                result["predicted_option"] = None
                results.append(result)
                skipped += 1
                continue
           
            if isinstance(videos, list):
                if not multi_image:
                    video_path = os.path.join(video_base_path, videos[0])

                    if not os.path.exists(video_path):
                        print(f"Video not found: {video_path}")
                        result["status"] = "video_not_found"
                        result["predicted_option"] = None
                        results.append(result)
                        skipped += 1
                        continue

                    # Extract frames
                    frames = extract_frames_from_video(video_path, num_frames=num_frames,strategy=frame_strategy)

                else:
                    frames = extract_and_concat_frames(videos, strategy=frame_strategy)
                    video_path = videos

            else:
                video_path = os.path.join(video_base_path, videos)
           

           
            if not frames:
                print(f"Could not extract frames from: {video_path}")
                result["status"] = "frame_extraction_failed"
                result["predicted_option"] = None
                results.append(result)
                skipped += 1
                continue
           
            image = frames[0]
            # image.save('/workspace/dilek/medgemma-challenge/img_' + str(total) + '.png')
           
            # Format prompt
            prompt = format_multiple_choice_prompt(sample, include_report=include_report, multi_image=multi_image)
           
            # Run inference
            response = run_inference(model, processor, image, prompt)
           
            # Extract answer
            predicted_letter = extract_answer_letter(response)

            result["model_response"] = response
            result["predicted_option"] = predicted_letter
            # result["predicted_option_regex"] = predicted_text
            result["status"] = "success"
            result["video_path"] = video_path
            result['prompt'] = prompt
           
            # Check if correct
            if predicted_letter and sample.get("correct_option"):
                total += 1
                if predicted_letter == sample["correct_option"]:
                    correct += 1
                    result["is_correct"] = True
                else:
                    result["is_correct"] = False
           
        except Exception as e:
            print(f"Error processing sample {sample.get('messages_id')}: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            result["predicted_option"] = None
            skipped += 1
       
        results.append(result)
       
        # Print running accuracy
        if total>1 and total % 50 == 0:
            print(f"Running accuracy: {correct}/{total} = {correct/total:.2%}")
   
    # Calculate final metrics
    metrics = {
        "total_samples": len(dataset),
        "processed": total,
        "skipped": skipped,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
    }
   
    # Save results
    output = {
        "metrics": metrics,
        "results": results,
    }
   
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
   
    print(f"\n{'='*50}")
    print(f"Evaluation Complete!")
    print(f"{'='*50}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Processed: {metrics['processed']}")
    print(f"Skipped: {metrics['skipped']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Results saved to: {output_path}")
   
    return metrics, results
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Run MedGemma on echocardiogram video dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the JSON dataset file"
    )
    parser.add_argument(
        "--video-base-path",
        type=str,
        required=True,
        help="Base path where video files are stored"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="medgemma_results.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/medgemma-1.5-4b-it",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--include-report",
        action="store_true",
        help="Include clinical report in the prompt"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of frames to extract from each video"
    )
    parser.add_argument(
        "--frame-strategy",
        type=str,
        default="middle",
        choices=["middle", "first", "uniform"],
        help="Strategy for selecting frames"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--multi-image",
        action="store_true",
        help="Flag to enable inputting multi-image"
    )
   
    args = parser.parse_args()
   
    # Load model
    print(30*'-' + 'loading model ' + args.model_id + 30*'-')
    model, processor = load_model(args.model_id)
   
    # Run evaluation
    evaluate_dataset(
        dataset_path=args.dataset,
        video_base_path=args.video_base_path,
        model=model,
        processor=processor,
        output_path=args.output,
        include_report=args.include_report,
        num_frames=args.num_frames,
        frame_strategy=args.frame_strategy,
        max_samples=args.max_samples,
        multi_image= args.multi_image,
    )
 
 
if __name__ == "__main__":
    main()