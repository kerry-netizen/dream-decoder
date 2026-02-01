"""
Cross-dream thread detection and meta-analysis engine.
Automatically identifies recurring patterns, symbols, and narrative arcs across multiple dreams.
"""

import json
from collections import Counter
from typing import List, Dict, Any, Tuple
from openai import OpenAI

client = OpenAI()


def extract_symbols_from_dreams(dreams: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract and count all symbols across dreams."""
    symbol_counts = Counter()

    for dream in dreams:
        analysis = dream.get("analysis", {})
        detected_keywords = analysis.get("detected_keywords", [])

        for keyword in detected_keywords:
            symbol_counts[keyword] += 1

    return dict(symbol_counts)


def extract_emotions_from_dreams(dreams: List[Dict[str, Any]]) -> Counter:
    """Extract and count emotions from dreams."""
    emotion_counts = Counter()

    for dream in dreams:
        # From input
        if dream.get("felt_during"):
            emotion_counts[dream["felt_during"]] += 1
        if dream.get("felt_after"):
            emotion_counts[dream["felt_after"]] += 1

        # From analysis
        analysis = dream.get("analysis", {})
        emotional_arc = analysis.get("emotional_arc", [])
        for stage in emotional_arc:
            emotion = stage.get("emotion", "")
            if emotion:
                emotion_counts[emotion] += 1

    return emotion_counts  # Return Counter to preserve .most_common()


def detect_recurring_patterns(
    dreams: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze multiple dreams to find recurring patterns.
    Returns structured data about patterns found.
    """
    if len(dreams) < 2:
        return {
            "recurring_symbols": [],
            "emotional_patterns": {},
            "narrative_themes": [],
        }

    # Extract symbols and emotions
    symbol_counts = extract_symbols_from_dreams(dreams)
    emotion_counts = extract_emotions_from_dreams(dreams)

    # Find recurring symbols (appear in 2+ dreams)
    recurring_symbols = [
        {"symbol": symbol, "count": count}
        for symbol, count in symbol_counts.items()
        if count >= 2
    ]
    recurring_symbols.sort(key=lambda x: x["count"], reverse=True)

    # Find dominant emotions
    dominant_emotions = [
        {"emotion": emotion, "count": count}
        for emotion, count in emotion_counts.most_common(5)
    ]

    # Collect narrative patterns from individual analyses
    narrative_themes = []
    for dream in dreams:
        analysis = dream.get("analysis", {})
        pattern = analysis.get("narrative_pattern", {})
        if pattern:
            narrative_themes.append(pattern)

    return {
        "recurring_symbols": recurring_symbols[:10],  # Top 10
        "emotional_patterns": dominant_emotions,
        "narrative_themes": narrative_themes,
        "total_dreams_analyzed": len(dreams),
    }


def call_thread_detection_llm(
    dreams: List[Dict[str, Any]],
    patterns: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use LLM to identify meaningful threads across dreams.
    Returns threads with names, descriptions, and dream IDs.
    """
    # Build prompt with dream summaries
    dream_summaries = []
    for i, dream in enumerate(dreams):
        analysis = dream.get("analysis", {})
        dream_summaries.append({
            "dream_id": dream["id"],
            "index": i + 1,
            "title": dream.get("title", "Untitled"),
            "date": dream.get("timestamp", ""),
            "summary": analysis.get("summary", ""),
            "key_symbols": [s.get("symbol", "") for s in analysis.get("key_symbols", [])[:5]],
            "felt_during": dream.get("felt_during", ""),
            "felt_after": dream.get("felt_after", ""),
        })

    system_prompt = """
You are a dream pattern analyst. You identify meaningful threads and recurring themes across multiple dreams.

A "thread" is a coherent narrative or thematic pattern that connects 2+ dreams. Examples:
- Recurring situations (e.g., being lost in buildings, water-related scenarios)
- Emotional arcs (e.g., escalating anxiety, recurring relief)
- Symbol clusters (e.g., animals + guidance, houses + transformation)
- Narrative patterns (e.g., pursuit dreams, threshold-crossing dreams)

Return VALID JSON with this structure:
{
  "threads": [
    {
      "thread_name": "Brief name (2-5 words)",
      "description": "2-3 sentence explanation of the thread",
      "dream_ids": [list of dream_id values that are part of this thread],
      "recurring_symbols": ["symbol1", "symbol2"],
      "emotional_pattern": "Description of emotional progression",
      "narrative_arc": "Description of how the narrative develops across dreams"
    }
  ]
}

Only identify threads that appear in 2+ dreams. If no clear threads exist, return empty threads array.
"""

    payload = {
        "dreams": dream_summaries,
        "detected_patterns": patterns,
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            timeout=30,
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Thread detection error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"threads": []}


def call_meta_analysis_llm(
    dreams: List[Dict[str, Any]],
    patterns: Dict[str, Any],
    threads: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Use LLM to generate high-level meta-analysis across all dreams.
    Returns insights about the dreamer's overall patterns.
    """
    # Build comprehensive summary
    dream_summaries = []
    for dream in dreams:
        analysis = dream.get("analysis", {})
        dream_summaries.append({
            "title": dream.get("title", "Untitled"),
            "date": dream.get("timestamp", ""),
            "summary": analysis.get("summary", ""),
            "micronarrative": analysis.get("micronarrative", ""),
        })

    system_prompt = """
You are a meta-dream analyst. You synthesize patterns across a person's dream journal to identify:
1. Overarching themes and preoccupations
2. Emotional trajectory over time
3. Recurring symbolic language unique to this dreamer
4. Psychological themes (identity, relationships, change, etc.)
5. Potential areas of growth or integration

Return VALID JSON:
{
  "overall_theme": "1-2 sentence overarching theme",
  "key_insights": ["insight 1", "insight 2", "insight 3"],
  "emotional_trajectory": "Description of emotional patterns over time",
  "symbolic_vocabulary": ["symbol1: meaning", "symbol2: meaning"],
  "psychological_themes": ["theme1", "theme2"],
  "areas_of_focus": "What these dreams seem to be working through",
  "integration_suggestions": ["suggestion1", "suggestion2"]
}

Be reflective, evidence-based, and avoid mysticism. Focus on what the dreams reveal about the dreamer's inner life.
"""

    payload = {
        "total_dreams": len(dreams),
        "dream_summaries": dream_summaries[-20:],  # Last 20 dreams
        "recurring_patterns": patterns,
        "identified_threads": [
            {
                "name": t.get("thread_name", ""),
                "description": t.get("description", ""),
            }
            for t in threads
        ],
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            timeout=30,
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Meta-analysis error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {
            "overall_theme": "Unable to generate meta-analysis",
            "key_insights": [],
            "emotional_trajectory": "",
            "symbolic_vocabulary": [],
            "psychological_themes": [],
            "areas_of_focus": "",
            "integration_suggestions": []
        }


def analyze_dream_threads(dreams: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Complete thread and meta-analysis pipeline.
    Returns (threads, meta_analysis).
    """
    if len(dreams) < 5:
        return [], {}

    # Step 1: Detect recurring patterns
    patterns = detect_recurring_patterns(dreams)

    # Step 2: Identify threads using LLM
    thread_data = call_thread_detection_llm(dreams, patterns)
    threads = thread_data.get("threads", [])

    # Step 3: Generate meta-analysis
    meta_analysis = call_meta_analysis_llm(dreams, patterns, threads)

    # Add patterns to meta-analysis
    meta_analysis["recurring_patterns"] = patterns

    return threads, meta_analysis
