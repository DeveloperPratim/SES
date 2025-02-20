# coding: utf-8
import aiohttp
import asyncio
import hashlib
import json
import random
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any
from googlesearch import search
import nltk
nltk.download('punkt_tab')
# Ensure NLTK resources are downloaded for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading 'punkt' package...")
    nltk.download('punkt')

def get_useragent():
    _useragent_list = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.62',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0'
    ]
    return random.choice(_useragent_list)

def google_search(query):
    """Perform Google search and return the list of URLs."""
    links = []
    try:
        # Perform the Google search request
        for url in search(query, lang='en'):
            links.append(url)  # Append each URL to the links list
    except Exception as e:
        print(f"Error during the search request: {e}")
    return links

async def fetch_web_content(url: str, session: aiohttp.ClientSession) -> str:
    """Fetch and clean the main content of a webpage."""
    try:
        headers = {"User-Agent": get_useragent()}
        async with session.get(url, headers=headers, timeout=5) as response:
            if response.status == 403:
                print(f"Access denied for {url}")
                return ""
            content = await response.text(errors='ignore')
            return content
    except aiohttp.ClientError as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

def get_shingles(text: str, k: int = 5) -> set:
    """Generate k-shingles (sets of k consecutive words) for a given text."""
    words = text.split()
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = " ".join(words[i:i + k])
        shingle_hash = hashlib.md5(shingle.encode("utf-8")).hexdigest()
        shingles.add(shingle_hash)
    return shingles

def similarity1(set1, set2):
    """Calculate the similarity score based on intersection of shingles."""
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    intersection = len(set1.intersection(set2))
    return (intersection / len(set1)) * 100

async def check_sentence_plagiarism(sentence: str) -> Dict[str, Any]:
    """Check plagiarism for a single sentence by searching and comparing content."""
    result = {"sentence": sentence, "matches": []}
    urls = google_search(sentence)  # Call google_search synchronously

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_web_content(url, session) for url in urls]
        contents = await asyncio.gather(*tasks)

    for url, content in zip(urls, contents):
        if content:
            original_shingles = get_shingles(sentence)
            content_shingles = get_shingles(content)
            similarity = similarity1(original_shingles, content_shingles)
            result["matches"].append({"url": url, "score": similarity})

    # Sort matches by score in descending order and take the highest
    if result["matches"]:
        result["matches"].sort(key=lambda x: x["score"], reverse=True)
        highest_match = result["matches"][0]
        result["highest_match"] = {"url": highest_match["url"], "score": highest_match["score"]}
    else:
        result["highest_match"] = {"url": None, "score": 0.0}

    return result

async def plagiarism_checker(text: str) -> List[Dict[str, Any]]:
    """Run plagiarism check on each sentence in the input text."""
    sentences = sent_tokenize(text)
    results = await asyncio.gather(*[check_sentence_plagiarism(sentence) for sentence in sentences])
    return get_score(results)
def get_score(data):
    """Generate a final plagiarism score, excluding scores less than 30."""
    result = []
    total_score = 0
    max_score = 0
    max_score_url = ""
    total_sentences = 0

    for entry in data:
        url = entry['highest_match']['url']
        score = entry['highest_match']['score']
        sentence = entry['sentence']

        # Exclude sentences with score less than 30
        if score < 30:
            plagiarism = False  # Mark as non-plagiarized
        else:
            plagiarism = True  # Mark as plagiarized if score >= 30

        # If the score is 30 or higher, add it to total_score for average calculation
        if score >= 30:
            total_score += score
            total_sentences += 1  # Count only plagiarized sentences for average

        # Track maximum score and corresponding URL
        if score > max_score:
            max_score = score
            max_score_url = url

        # Check if we should merge with the last entry
        if result and result[-1]['url'] == url and result[-1]['score'] == score:
            result[-1]['sentence'] += " " + sentence  # Append the sentence
        else:
            # Add a new entry with plagiarism flag
            result.append({
                'sentence': sentence,
                'url': url,
                'score': score,
                'plagiarism': plagiarism
            })

    # Calculate average score only from sentences with score >= 30
    average_score = total_score / total_sentences if total_sentences else 0

    # Create the final structured response
    response = {
        "plagiarism": any(entry['plagiarism'] for entry in result),
        "average_score": average_score,
        "max_score": max_score,
        "max_score_url": max_score_url,
        "data": result
    }

    return response

# Example input text for plagiarism checking
# input_text = """
# An Operating System can be defined as an interface between user and hardware. It is responsible for the execution of all the processes, Resource Allocation, CPU scheduling, memory management, and providing security. The OS also manages devices such as printers, monitors, and storage devices. Without an operating system, a computer cannot function effectively.
# """
# Running the plagiarism checker asynchronously
# results_plagiarism = await plagiarism_checker(input_text)

# Printing the results in a readable format
# print(json.dumps(results_plagiarism, indent=2))






from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def detect_ai_generated_text_advanced_v3(
    text,
    model_name="roberta-base-openai-detector",
    threshold=0.5,
    max_length=512
):
    """
    Advanced detection of AI-generated text with detailed metrics, AI flag, and percentage.

    Parameters:
        text (str): Input text to analyze.
        model_name (str): Pre-trained model to use for detection.
        threshold (float): Confidence threshold for categorization (default: 0.5).
        max_length (int): Maximum length for tokenization (default: 512).

    Returns:
        dict: Detailed output with categorization, confidence, metrics, AI flag, and percentage.
    """
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Tokenize and prepare text
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        token_count = len(tokens["input_ids"][0])

        # Inference
        outputs = model(**tokens)
        probabilities = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
        ai_confidence = probabilities[1]
        human_confidence = probabilities[0]
        category = "AI-generated" if ai_confidence > threshold else "Human-written"

        # AI flag and percentage
        is_ai = ai_confidence > threshold
        ai_percentage = round(ai_confidence * 100, 2)

        # Additional metrics
        char_count = len(text)
        word_count = len(text.split())
        avg_word_length = char_count / word_count if word_count > 0 else 0
        confidence_diff = abs(ai_confidence - human_confidence)

        return {
            "category": category,
            "AI": is_ai,
            "AI_Percentage": ai_percentage,
            "confidence": ai_confidence if is_ai else human_confidence,
            "confidence_difference": confidence_diff,
            "probabilities": {
                "Human-written": human_confidence,
                "AI-generated": ai_confidence
            },
            "metrics": {
                "character_count": char_count,
                "word_count": word_count,
                "average_word_length": round(avg_word_length, 2),
                "token_count": token_count,
                "max_token_length": max_length
            },
            "threshold_used": threshold
        }
    except Exception as e:
        return {"error": str(e)}

# Example usage
#text = "In a world driven by rapid technological advancement, artificial intelligence has emerged as a transformative force. From revolutionizing healthcare with predictive diagnostics to enhancing efficiency in industries through automation, AI continues to redefine the boundaries of what is possible. However, this unprecedented growth also poses ethical dilemmas, emphasizing the need for responsible innovation to ensure these technologies benefit humanity as a whole."
#result_AI = detect_ai_generated_text_advanced_v3(text, threshold=0.7)
#print(result)
#print()




import re
import math
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load model
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
device = "cuda" if torch.cuda.is_available() else "cpu"

if 'model' not in globals():
    model = AutoAWQForCausalLM.from_quantized(
        model_name_or_path,
        fuse_layers=True,
        trust_remote_code=False,
        safetensors=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)

models_loaded = True

# Function to evaluate the student's answer
def evaluate_answer(question, answer, full_marks=5, expected_answer=None):
    # Basic Information

    prompt = f"""
    Question: {question}
    Answer: {answer}
    Full Marks: {full_marks}
    """

    if expected_answer:
        prompt += f"Expected Answer: {expected_answer}\n"

    # Evaluation instructions
    prompt += """
    Evaluate based on strictly: Correctness of answer, Completeness of answer, Clarity, Depth, and Relevance of answer , if answer not according to asked question or giving nearly similar answer just give it a 0 marks.

    Return JSON with:
    - marks_obtained
    - feedback  ( detailed overall feedback )
    - strengths ( detailed positive aspects )
    - areas_for_improvement ( for getting full marks)
    - clarity_score (1-10)
    - depth_score (1-10)
    """


    # Tokenize the prompt for the model
    tokens = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    # Generate the model's response with the given prompt
#    generation_output = model.generate(
#        tokens, do_sample=True, temperature=0.7, top_p=0.95, top_k=40, max_new_tokens=512
#    )
    generation_output = model.generate(
        tokens,
        do_sample=False,          # Deterministic output for faster generation
        max_new_tokens=256,       # Lowered for faster response while sufficient for JSON output
        early_stopping=True       # Stops generation once output is complete
    )

    # Decode the output from the model
    response = tokenizer.decode(generation_output[0], skip_special_tokens=True)

    return response


def round_based_on_fraction(number):
    # Get the fractional part of the number
    fractional_part = number - math.floor(number)

    # If the fractional part is greater than 0.5, round up, else round down
    if fractional_part > 0.5:
        return math.ceil(number)  # Round up
    else:
        return math.floor(number)  # Round down




import re

def extract_score_and_feedback(generated_text):
    """
    Extract detailed evaluation information from the generated text using regular expressions.

    :param generated_text: The raw output from the model.
    :return: A dictionary with extracted evaluation details.
    """
    # Define the patterns for each field
    score_pattern = r'"marks_obtained":\s*([0-9.]+)'  # Extract numerical score
    feedback_pattern = r'"feedback":\s*"([^"]+)"'  # Extract feedback text, handling newlines
    strengths_pattern = r'"strengths":\s*\[([^\]]+)\]'  # Extract strengths list
    areas_for_improvement_pattern = r'"areas_for_improvement":\s*\[([^\]]+)\]'  # Extract areas for improvement list
    clarity_score_pattern = r'"clarity_score":\s*([0-9.]+)'  # Extract clarity score
    depth_score_pattern = r'"depth_score":\s*([0-9.]+)'  # Extract depth score

    # Find matches using regular expressions
    score_match = re.search(score_pattern, generated_text)
    feedback_match = re.search(feedback_pattern, generated_text)
    strengths_match = re.search(strengths_pattern, generated_text)
    areas_for_improvement_match = re.search(areas_for_improvement_pattern, generated_text)
    clarity_score_match = re.search(clarity_score_pattern, generated_text)
    depth_score_match = re.search(depth_score_pattern, generated_text)

    # Extract values or set them to None if not found
    score = float(score_match.group(1)) if score_match else None
    feedback = feedback_match.group(1).strip() if feedback_match else None  # Strip to remove extra spaces/newlines

    # Handle strengths and areas for improvement (could be a list or a single string)
    strengths = parse_list_or_string(strengths_match.group(1)) if strengths_match else []
    areas_for_improvement = parse_list_or_string(areas_for_improvement_match.group(1)) if areas_for_improvement_match else []

    clarity_score = float(clarity_score_match.group(1)) if clarity_score_match else None
    depth_score = float(depth_score_match.group(1)) if depth_score_match else None

    # Return the extracted data as a dictionary
    return {
        "marks_obtained": score,
        "feedback": feedback,
        "strengths": strengths,
        "areas_for_improvement": areas_for_improvement,
        "clarity_score": clarity_score,
        "depth_score": depth_score
    }

def parse_list_or_string(value):
    """
    Parse a field that could either be a list of strings or a single string.
    If it's a list, it will return a list; if it's a single string, it will return a list with one item.

    :param value: The raw string to be parsed (could be a list or a string).
    :return: A list of strings.
    """
    # If the value looks like a list (contains commas), split it into a list
    if isinstance(value, str) and "," in value:
        # Remove leading/trailing whitespace and split by commas
        return [item.strip().strip('"') for item in value.split(",")]
    # Otherwise, return a list containing the value as a single item
    elif isinstance(value, str):
        return [value.strip().strip('"')]
    return []


#################################################
#################################################
################################################# GetData
#################################################
#################################################
#################################################





import asyncio
import math
import re

async def GetData(question, student_answer, expected_answer, marks):
    if not models_loaded:
        raise RuntimeError("Models are not loaded. Call `load_models()` first.")

    # Ensure non-empty inputs
    student_answer = student_answer or ""  # Default to empty string if None or empty
    expected_answer = expected_answer or ""  # Default to empty string if None or empty
    marks = marks or 0  # Default to 0 if marks are None or empty
    question = question or ""  # Default to empty string if None or empty

    # Generate response using the model
    generated_text = evaluate_answer(question, student_answer, marks, expected_answer)

    # Extract score and feedback from the model's output
    result = extract_score_and_feedback(generated_text)

    # Handle missing or None values gracefully using .get() with default values
    score = result.get('marks_obtained', 0)  # Default to 0 if score is None or not found
    score_old = score
    feedback = result.get('feedback', "")  # Default to an empty string if feedback is None or not found
    strengths = result.get('strengths', "")  # Default to an empty string if strengths are None or not found
    areas_for_improvement = result.get('areas_for_improvement', "")  # Default to an empty string if areas are None or not found
    clarity_score = result.get('clarity_score', 0)  # Default to 0 if clarity score is None or not found
    depth_score = result.get('depth_score', 0)  # Default to 0 if depth score is None or not found

    # Ensure score is not None before using it in arithmetic operations
    score = score if score is not None else 0

    # Logarithmic scaling of score
    log_score = math.log(score + 1) if score is not None else 0  # Logarithm of score (score + 1 to avoid log(0))
    weighted_score = log_score * 0.5  # Adjust weight as needed

    # Calculate initial marks based on the weighted sum
    final_marks_obtained = weighted_score * marks  # Scale by full marks

    # Initialize penalty and reason
    penalty = 0.0
    penalty_reasons = []

    # Apply penalty if marks exceed 90% of full marks
    max_allowed_marks = marks * 0.9
    if final_marks_obtained > max_allowed_marks:
        penalty_factor = 0.8  # Apply a 20% penalty
        final_marks_obtained *= penalty_factor
        penalty += 0.2  # Penalty is 20% of the score
        penalty_reasons.append("Penalty applied for exceeding 90% of full marks")

    # Check plagiarism and AI detection
    results_plagiarism = await plagiarism_checker(student_answer) or {}
    result_AI = detect_ai_generated_text_advanced_v3(student_answer, threshold=0.7) or {}

    if results_plagiarism.get("plagiarism", False) and results_plagiarism.get("average_score", 0.0) > 0.3:
        penalty += 0.25  # Apply a 25% penalty for plagiarism
        penalty_reasons.append("Penalty applied for plagiarism exceeding 30%")

    if result_AI.get("AI", False) and result_AI.get("AI_Percentage", 0.0) > 30:
        penalty += 0.25  # Apply a 25% penalty for AI-generated text exceeding 30%
        penalty_reasons.append("Penalty applied for AI detection exceeding 30%")

    # Apply total penalty and adjust marks
    final_marks_obtained = final_marks_obtained * (1 - penalty)

    # Ensure marks are never below 0
    final_marks_obtained = max(final_marks_obtained, 0)

    # Analyze student's answer
    student_word_count = len(student_answer.split())  # Count words in student's answer
    student_sentence_count = len(re.split(r'[.!?]+', student_answer.strip())) - 1  # Count sentences

    # Prepare final result dictionary
    result_data = {
        "full_marks": marks,
        "obtained_marks": score_old,
        "strengths": strengths,
        "areas_for_improvement": areas_for_improvement,
        "clarity_score": clarity_score,
        "depth_score": depth_score,
        "final_marks_obtained": round_based_on_fraction(final_marks_obtained),
        "feedback": feedback,
        "student_word_count": student_word_count,
        "student_sentence_count": student_sentence_count,
        "plagiarism": results_plagiarism.get("plagiarism", False),
        "average_score": results_plagiarism.get("average_score", 0.0),
        "max_score": results_plagiarism.get("max_score", 0.0),
        "max_score_url": results_plagiarism.get("max_score_url", ""),
        "AI_category": result_AI.get("category", "Unknown"),
        "AI_flag": result_AI.get("AI", False),
        "AI_percentage": result_AI.get("AI_Percentage", 0.0),
        "AI_probabilities": result_AI.get("probabilities", {}),
        "reason": "; ".join(penalty_reasons),  # List of reasons for penalties applied
    }

    return result_data
