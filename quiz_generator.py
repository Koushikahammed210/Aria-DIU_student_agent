# quiz_generator.py — Quiz JSON parser + template engine for Aria
# Handles: AI output parsing, validation, template-based fallback

import re
import json
import random


def parse_quiz_json(ai_response: str) -> list:
    """
    Parse AI response into a list of quiz question dicts.
    Robust parser that handles various AI output formats:
    - Clean JSON array
    - JSON wrapped in markdown code fences
    - JSON with trailing text
    - Partially valid JSON
    
    Returns: list of validated question dicts
    """
    if not ai_response:
        return []
    
    # Try multiple extraction strategies
    candidates = []
    
    # Strategy 1: Direct JSON parse
    candidates.append(ai_response.strip())
    
    # Strategy 2: Extract from code fences
    code_match = re.search(r'```(?:json)?\s*(.*?)\s*```', ai_response, re.DOTALL)
    if code_match:
        candidates.insert(0, code_match.group(1))
    
    # Strategy 3: Find JSON array boundaries
    start = ai_response.find('[')
    end = ai_response.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidates.insert(0, ai_response[start:end+1])
    
    # Try each candidate
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, list) and len(data) > 0:
                validated = _validate_questions(data)
                if validated:
                    return validated
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            fixed = _fix_json(candidate)
            if fixed:
                try:
                    data = json.loads(fixed)
                    if isinstance(data, list) and len(data) > 0:
                        validated = _validate_questions(data)
                        if validated:
                            return validated
                except json.JSONDecodeError:
                    continue
    
    # Strategy 4: Regex extraction of individual question objects
    return _regex_extract_questions(ai_response)


def _validate_questions(questions: list) -> list:
    """Validate and normalize a list of question dicts."""
    validated = []
    required_keys = {"question", "A", "B", "C", "D", "correct"}
    
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        
        # Check required keys
        if not required_keys.issubset(q.keys()):
            continue
        
        # Normalize correct answer to uppercase single letter
        correct = str(q["correct"]).strip().upper()
        if correct not in ("A", "B", "C", "D"):
            continue
        
        # Build validated question
        validated_q = {
            "id": i + 1,
            "question": str(q["question"]).strip(),
            "A": str(q["A"]).strip(),
            "B": str(q["B"]).strip(),
            "C": str(q["C"]).strip(),
            "D": str(q["D"]).strip(),
            "correct": correct,
            "explanation": str(q.get("explanation", "No additional explanation available.")).strip(),
            "difficulty": str(q.get("difficulty", "medium")).strip().lower(),
        }
        
        # Skip if question or options are empty
        if not validated_q["question"] or not validated_q["A"]:
            continue
        
        validated.append(validated_q)
    
    return validated


def _fix_json(text: str) -> str:
    """Try to fix common JSON issues in AI output."""
    # Remove trailing commas before ] or }
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Remove JavaScript-style comments
    text = re.sub(r'//.*?\n', '\n', text)
    # Fix missing quotes around KEYS only (not inside string values)
    # Capture the opening char [{, in group 1, then quote the key in group 2
    # This avoids variable-width lookbehind which Python re doesn't support
    text = re.sub(r'([{,\[]\s*)(\w+)\s*:', r'\1"\2":', text)
    return text


def _regex_extract_questions(text: str) -> list:
    """
    Last resort: extract questions using regex patterns.
    Handles non-JSON AI output like numbered lists.
    """
    questions = []
    
    # Split by question patterns
    q_blocks = re.split(r'(?:^|\n)(?:\d+[\.\)]\s*|(?:Q\d+:))', text)
    
    for block in q_blocks:
        block = block.strip()
        if not block or len(block) < 20:
            continue
        
        # Try to extract question and options
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) < 3:
            continue
        
        question_text = lines[0].rstrip('?') + '?'
        options = {}
        correct = None
        
        for line in lines[1:]:
            # Match "A) option" or "A. option" or "A: option" patterns
            opt_match = re.match(r'^([A-D])[\)\.\:]\s*(.+)', line, re.IGNORECASE)
            if opt_match:
                letter = opt_match.group(1).upper()
                options[letter] = opt_match.group(2).strip()
                continue
            
            # Check for correct answer indicator
            if 'correct' in line.lower() or 'answer' in line.lower():
                ans_match = re.search(r'([A-D])', line, re.IGNORECASE)
                if ans_match:
                    correct = ans_match.group(1).upper()
        
        if len(options) == 4 and question_text:
            if not correct:
                correct = "A"  # Default
            
            questions.append({
                "id": len(questions) + 1,
                "question": question_text,
                "A": options.get("A", "Option A"),
                "B": options.get("B", "Option B"),
                "C": options.get("C", "Option C"),
                "D": options.get("D", "Option D"),
                "correct": correct,
                "explanation": "Review the source material for more details on this topic.",
                "difficulty": "medium",
            })
    
    return questions


# ══════════════════════════════════════════════════════════════════════════════
#  Template Engine — Zero AI fallback (pure Python)
# ══════════════════════════════════════════════════════════════════════════════

def generate_template_quiz(content: str, topic: str, difficulty: str, count: int) -> list:
    """
    Generate quiz questions WITHOUT any AI model.
    Uses text extraction and sentence transformation.
    Works 100% offline, no GPU, no internet.
    """
    if content and len(content) > 50:
        return _file_based_quiz(content, difficulty, count)
    else:
        return _topic_based_quiz(topic, difficulty, count)


def _file_based_quiz(content: str, difficulty: str, count: int) -> list:
    """Generate fill-in-the-blank MCQs from file content."""
    # Extract key sentences (ones with important terms)
    sentences = _extract_key_sentences(content)
    
    if not sentences:
        return _topic_based_quiz("General Knowledge", difficulty, count)
    
    # Get all significant words from content for distractors
    all_terms = _extract_terms(content)
    
    questions = []
    for i, sentence in enumerate(sentences[:count]):
        # Find the keyword to blank out
        keyword = _extract_keyword_from_sentence(sentence, all_terms)
        
        if not keyword:
            continue
        
        # Create fill-in-the-blank
        question_text = f"Fill in the blank: {sentence.replace(keyword, '______')}"
        
        # Generate wrong options from other terms
        wrong_options = [t for t in all_terms if t != keyword and len(t) > 2][:3]
        
        while len(wrong_options) < 3:
            wrong_options.append(f"None of the above ({'XYZ'[len(wrong_options)]})")
        
        # Shuffle options
        options = [keyword] + wrong_options[:3]
        random.shuffle(options)
        correct_letter = ["A", "B", "C", "D"][options.index(keyword)]
        
        questions.append({
            "id": i + 1,
            "question": question_text,
            "A": options[0],
            "B": options[1],
            "C": options[2],
            "D": options[3],
            "correct": correct_letter,
            "explanation": f"The correct answer is '{keyword}'. From the source material: {sentence}",
            "difficulty": difficulty,
        })
    
    return questions


def _topic_based_quiz(topic: str, difficulty: str, count: int) -> list:
    """Generate generic quiz questions based on a topic name."""
    templates = [
        {
            "q": f"What is the primary definition of {topic}?",
            "A": f"A fundamental concept in {topic} studies",
            "B": "An unrelated scientific phenomenon",
            "C": "A type of literary device",
            "D": "A mathematical constant",
            "correct": "A",
            "explanation": f"This is the most accurate description of {topic} among the given options. {topic} is a well-established field/concept with its own principles and methodologies.",
        },
        {
            "q": f"Which of the following best describes the importance of {topic}?",
            "A": "It has no practical applications",
            "B": f"It provides essential frameworks and tools for understanding related phenomena",
            "C": "It is only relevant in theoretical contexts",
            "D": "It has been completely replaced by newer approaches",
            "correct": "B",
            "explanation": f"{topic} remains highly relevant because it provides foundational knowledge and practical tools that are widely used across multiple domains.",
        },
        {
            "q": f"What is a common misconception about {topic}?",
            "A": f"That {topic} is only for experts",
            "B": f"That {topic} is widely understood by everyone",
            "C": f"That {topic} is a simple subject with no depth",
            "D": f"That {topic} has no real-world applications",
            "correct": "C",
            "explanation": f"Many people underestimate the complexity of {topic}. While the basics may seem straightforward, {topic} has significant depth and nuance that requires careful study.",
        },
        {
            "q": f"Which approach is most effective for learning {topic}?",
            "A": "Memorizing definitions without context",
            "B": "Avoiding practical exercises entirely",
            "C": f"Combining theoretical study with hands-on practice in {topic}",
            "D": "Only reading about {topic} without applying concepts",
            "correct": "C",
            "explanation": f"Research shows that combining theory with practice is the most effective way to learn {topic}. This approach helps solidify understanding and develop practical skills.",
        },
        {
            "q": f"How has {topic} evolved over time?",
            "A": f"It has remained completely unchanged since its inception",
            "B": f"It has been abandoned by researchers and practitioners",
            "C": f"It has incorporated new discoveries and methodologies, becoming more comprehensive",
            "D": f"It has become simpler and less relevant",
            "correct": "C",
            "explanation": f"Like most fields, {topic} has evolved significantly, integrating new research findings, technologies, and methodologies to become more robust and applicable.",
        },
        {
            "q": f"Which skill is most important for mastering {topic}?",
            "A": "Physical strength and endurance",
            "B": f"Critical thinking and analytical reasoning",
            "C": "Musical ability",
            "D": "Artistic talent exclusively",
            "correct": "B",
            "explanation": f"Critical thinking and analytical reasoning are essential for {topic} because they allow practitioners to evaluate evidence, solve problems, and develop new insights.",
        },
        {
            "q": f"What role does {topic} play in modern society?",
            "A": "It has no role whatsoever",
            "B": "It is only used in academic settings",
            "C": f"It influences multiple sectors including education, technology, and industry",
            "D": "It is restricted to entertainment purposes",
            "correct": "C",
            "explanation": f"{topic} has broad societal impact, affecting how we approach problems and develop solutions across education, technology, industry, and many other sectors.",
        },
        {
            "q": f"What distinguishes {topic} from related fields?",
            "A": "Nothing — it is identical to all other fields",
            "B": f"Its unique focus, specialized methodologies, and distinct body of knowledge",
            "C": "It uses no methodologies at all",
            "D": "It is the only field that exists",
            "correct": "B",
            "explanation": f"{topic} has its own unique focus area and specialized approaches that set it apart from related fields, even though there may be some overlap in certain areas.",
        },
        {
            "q": f"Which statement about {topic} is most accurate?",
            "A": f"{topic} is a static field with no room for innovation",
            "B": f"{topic} continues to grow and adapt with new research and applications",
            "C": f"{topic} is only useful for a narrow set of problems",
            "D": f"{topic} has been proven to be incorrect",
            "correct": "B",
            "explanation": f"{topic} is a dynamic and evolving field. New research continuously expands our understanding and opens up new applications and possibilities.",
        },
        {
            "q": f"What is the best way to stay current with developments in {topic}?",
            "A": "Ignore all new information",
            "B": "Rely exclusively on outdated textbooks",
            "C": f"Follow recent publications, attend conferences, and engage with the {topic} community",
            "D": "Wait for someone to tell you about changes",
            "correct": "C",
            "explanation": f"Staying current in {topic} requires active engagement with the community, reading recent publications, and participating in professional development opportunities.",
        },
    ]
    
    # Select requested number of questions
    selected = templates[:min(count, len(templates))]
    
    # If we need more questions than templates, cycle through
    while len(selected) < count:
        idx = len(selected) % len(templates)
        q = templates[idx].copy()
        q["id"] = len(selected) + 1
        selected.append(q)
    
    # Add IDs and difficulty
    result = []
    for i, q in enumerate(selected[:count]):
        result.append({
            "id": i + 1,
            "question": q["q"],
            "A": q["A"],
            "B": q["B"],
            "C": q["C"],
            "D": q["D"],
            "correct": q["correct"],
            "explanation": q["explanation"],
            "difficulty": difficulty,
        })
    
    return result


# ── Text Processing Helpers for Template Engine ─────────────────────────────

def _extract_key_sentences(content: str, max_sentences: int = 30) -> list:
    """Extract sentences that contain key information (definitions, facts)."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    key_sentences = []
    priority_patterns = [
        r'is\s+(?:defined|described|known)\s+as',
        r'refers?\s+to',
        r'means?\s+that',
        r'is\s+a\s+(?:type|kind|form|method|process|concept)',
        r'consists?\s+of',
        r'includes?\s+(?:the|a|an)',
        r'used\s+(?:for|to|in|by)',
        r'important\s+(?:because|for|in)',
        r'plays?\s+a\s+(?:key|critical|vital|important)',
        r'required\s+(?:for|to|by)',
        r'involves?\s+(?:the|a|an)',
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20 or len(sentence) > 200:
            continue
        
        # Prioritize sentences with definition/fact patterns
        is_priority = any(re.search(p, sentence, re.IGNORECASE) for p in priority_patterns)
        
        if is_priority:
            key_sentences.insert(0, sentence)  # High priority first
        elif len(sentence.split()) >= 6:  # At least 6 words
            key_sentences.append(sentence)
        
        if len(key_sentences) >= max_sentences:
            break
    
    return key_sentences


def _extract_terms(content: str) -> list:
    """Extract significant terms/keywords from content for distractors."""
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'must', 'ought',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
        'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
        'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'where',
        'when', 'how', 'why', 'all', 'each', 'every', 'both', 'few',
        'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'because',
        'but', 'and', 'or', 'if', 'while', 'as', 'with', 'from', 'for',
        'about', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'also', 'of', 'in', 'to', 'on', 'at', 'by',
        'up', 'out', 'off', 'over', 'down', 'any', 'many', 'much',
    }
    
    # Tokenize and filter
    words = re.findall(r'\b[A-Za-z][A-Za-z\-]{3,}\b', content.lower())
    
    # Count frequency
    freq = {}
    for word in words:
        if word not in stop_words:
            freq[word] = freq.get(word, 0) + 1
    
    # Sort by frequency (most common terms first) — these are likely key terms
    terms = sorted(freq.keys(), key=lambda w: freq[w], reverse=True)
    
    # Also extract multi-word terms (capitalized phrases)
    multi_word = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content)
    terms = list(dict.fromkeys(multi_word + terms))  # Deduplicate, multi-word first
    
    return terms[:50]  # Top 50 terms


def _extract_keyword_from_sentence(sentence: str, all_terms: list) -> str:
    """Find the most important keyword in a sentence to blank out."""
    # Prefer multi-word terms
    for term in all_terms:
        if ' ' in term and term.lower() in sentence.lower():
            return term
    
    # Then single-word terms
    words = sentence.split()
    for term in all_terms:
        if term.lower() in [w.lower().strip('.,;:!?') for w in words]:
            return term
    
    # Fallback: pick the longest non-stop word
    stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'this', 'that', 'these', 'those', 'which', 'what',
            'who', 'whom', 'whose', 'where', 'when', 'how', 'why', 'and', 'but', 'or',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as'}
    
    candidates = [w.strip('.,;:!?') for w in words if w.lower() not in stop and len(w) > 3]
    if candidates:
        return max(candidates, key=len)
    
    return ""
