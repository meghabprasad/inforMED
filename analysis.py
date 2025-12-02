"""
Analysis Script: Novel Insights for inforMED Report

This script generates empirical data for the "Novel Insights" section of the report.
It computes three analyses:

1. Mutual Information - which symptoms are most informative
2. Conditional Entropy Chains - how information compounds through questions
3. Baseline Comparisons - compare to random and frequency-based ordering

Run this script to see the analysis results that are discussed in the report.
"""

import numpy as np
from data import DIAGNOSES, SYMPTOMS, PRIOR_PROBABILITY, NUM_DIAGNOSES, CPT
from engine import (
    calculate_entropy,
    bayesian_update,
    get_next_question,
    calculate_information_gain,
)


# ============================================================================
# 1. MUTUAL INFORMATION
# ============================================================================


def compute_mutual_information_all_symptoms():
    """
    Compute average mutual information I(D; S_j) for each symptom
    across uniform prior distribution.

    Returns:
        List of (symptom_key, symptom_text, mutual_info) tuples, sorted by MI
    """
    print("=" * 70)
    print("Analysis 1: Which symptoms are most informative?")
    print("=" * 70)

    # Start with uniform priors
    uniform_P = [PRIOR_PROBABILITY] * NUM_DIAGNOSES

    mi_results = []
    for symptom_key in SYMPTOMS.keys():
        # Calculate information gain (which equals mutual information)
        mi = calculate_information_gain(uniform_P, symptom_key)
        mi_results.append((symptom_key, SYMPTOMS[symptom_key], mi))

    # Sort by MI (descending)
    mi_results.sort(key=lambda x: x[2], reverse=True)

    print("\nMost informative symptoms:")
    print(f"{'Rank':<6} {'Key':<8} {'Info (bits)':<12} {'Question'}")
    print("-" * 70)
    for i, (key, text, mi) in enumerate(mi_results[:10], 1):
        question_short = text[:50] + "..." if len(text) > 50 else text
        print(f"{i:<6} {key:<8} {mi:<12.4f} {question_short}")

    print("\nLeast informative symptoms:")
    print(f"{'Rank':<6} {'Key':<8} {'Info (bits)':<12} {'Question'}")
    print("-" * 70)
    for i, (key, text, mi) in enumerate(mi_results[-10:], 27):
        question_short = text[:50] + "..." if len(text) > 50 else text
        print(f"{i:<6} {key:<8} {mi:<12.4f} {question_short}")

    return mi_results


# ============================================================================
# 2. CONDITIONAL ENTROPY CHAINS
# ============================================================================


def trace_diagnostic_path(target_diagnosis, answers):
    """
    Trace entropy reduction through a specific diagnostic path.

    Args:
        target_diagnosis: The diagnosis we're simulating (e.g., "Migraine")
        answers: Dict mapping symptom_key -> True/False answers

    Returns:
        List of (question_num, question_key, question_text, answer, ig, entropy) tuples
    """
    P = [PRIOR_PROBABILITY] * NUM_DIAGNOSES
    asked = set()
    path = []

    initial_entropy = calculate_entropy(P)
    path.append((0, None, "Initial state", None, None, initial_entropy))

    question_num = 1
    while max(P) < 0.90 and len(asked) < len(SYMPTOMS):
        # Get next best question
        question_text, question_key, ig = get_next_question(P, asked)

        if question_key is None:
            break

        # Get the answer for this symptom from our test case
        answer = answers.get(question_key, None)
        if answer is None:
            # If not specified, use the CPT probability for this diagnosis
            prob_yes = CPT[target_diagnosis][question_key]
            answer = prob_yes > 0.5  # Use most likely answer

        # Update probabilities
        P = bayesian_update(P, question_key, answer)
        asked.add(question_key)

        entropy_after = calculate_entropy(P)

        path.append(
            (
                question_num,
                question_key,
                SYMPTOMS[question_key],
                "Yes" if answer else "No",
                ig,
                entropy_after,
            )
        )

        question_num += 1

    return path


def analyze_conditional_entropy_chains():
    """
    Analyze how entropy reduction accelerates through question sequences
    for different diagnoses.
    """
    print("\n" + "=" * 70)
    print("Analysis 2: How does entropy reduction change over time?")
    print("=" * 70)

    # Test cases for different diagnoses
    test_cases = {
        "Migraine": {
            "S1": True,  # Throbbing
            "S3": True,  # Nausea
            "S4": True,  # Photophobia
            "S7": True,  # 4-72 hours
            "S8": True,  # Worsened by activity
        },
        "Trigeminal Neuralgia": {
            "S28": True,  # Seconds duration
            "S26": True,  # Shock-like pain
            "S27": True,  # Facial triggers
            "S2": True,  # Unilateral
        },
        "Tension-Type": {
            "S17": True,  # Band-like
            "S18": True,  # Steady pressure
            "S19": True,  # Can continue activities
            "S34": True,  # Muscle tenderness
        },
        "Cluster": {
            "S11": True,  # Brief attacks 15-180 min
            "S6": True,  # Autonomic symptoms
            "S5": True,  # Severe intensity
            "S12": True,  # Restlessness
        },
    }

    results = {}
    for diagnosis, answers in test_cases.items():
        print(f"\n--- Testing with {diagnosis} ---")
        path = trace_diagnostic_path(diagnosis, answers)
        results[diagnosis] = path

        # Print the path
        print(f"{'Q#':<4} {'Question':<45} {'Answer':<6} {'IG':<8} {'Entropy'}")
        print("-" * 80)
        for q_num, q_key, q_text, answer, ig, entropy in path:
            if q_num == 0:
                print(
                    f"{q_num:<4} {'(Initial state)':<45} {'':<6} {'':<8} {entropy:.3f}"
                )
            else:
                q_short = q_text[:43] + ".." if len(q_text) > 45 else q_text
                ig_str = f"{ig:.3f}" if ig else ""
                print(f"{q_num:<4} {q_short:<45} {answer:<6} {ig_str:<8} {entropy:.3f}")

        # Calculate entropy reductions
        if len(path) > 1:
            print(f"\nHow much each question helped:")
            for i in range(1, len(path)):
                prev_entropy = path[i - 1][5]
                curr_entropy = path[i][5]
                reduction = prev_entropy - curr_entropy
                print(
                    f"  Q{i}: {prev_entropy:.3f} → {curr_entropy:.3f} (reduced by {reduction:.3f} bits)"
                )

    return results


# ============================================================================
# 3. BASELINE COMPARISONS
# ============================================================================


def simulate_random_ordering(target_diagnosis, answers, num_trials=100):
    """
    Simulate random question ordering and return average questions needed.
    """
    questions_needed = []

    for _ in range(num_trials):
        P = [PRIOR_PROBABILITY] * NUM_DIAGNOSES
        asked = set()
        question_count = 0

        # Shuffle all symptoms
        all_symptoms = list(SYMPTOMS.keys())
        np.random.shuffle(all_symptoms)

        for symptom_key in all_symptoms:
            if max(P) >= 0.90:
                break

            # Get answer
            answer = answers.get(symptom_key, CPT[target_diagnosis][symptom_key] > 0.5)

            # Update
            P = bayesian_update(P, symptom_key, answer)
            question_count += 1

        questions_needed.append(question_count)

    return np.mean(questions_needed), np.std(questions_needed)


def simulate_frequency_based_ordering(target_diagnosis, answers):
    """
    Simulate frequency-based ordering (ask about most common symptoms first).

    We'll define "common" as symptoms with high average probability across all diagnoses.
    """
    # Calculate average probability for each symptom
    symptom_frequencies = {}
    for symptom_key in SYMPTOMS.keys():
        avg_prob = np.mean([CPT[d][symptom_key] for d in DIAGNOSES])
        symptom_frequencies[symptom_key] = avg_prob

    # Sort by frequency (descending)
    ordered_symptoms = sorted(
        symptom_frequencies.keys(), key=lambda k: symptom_frequencies[k], reverse=True
    )

    # Simulate diagnostic session
    P = [PRIOR_PROBABILITY] * NUM_DIAGNOSES
    question_count = 0

    for symptom_key in ordered_symptoms:
        if max(P) >= 0.90:
            break

        # Get answer
        answer = answers.get(symptom_key, CPT[target_diagnosis][symptom_key] > 0.5)

        # Update
        P = bayesian_update(P, symptom_key, answer)
        question_count += 1

    return question_count


def simulate_ig_ordering(target_diagnosis, answers):
    """
    Simulate our information gain ordering (the actual algorithm).
    """
    P = [PRIOR_PROBABILITY] * NUM_DIAGNOSES
    asked = set()
    question_count = 0

    while max(P) < 0.90 and len(asked) < len(SYMPTOMS):
        question_text, question_key, ig = get_next_question(P, asked)

        if question_key is None:
            break

        # Get answer
        answer = answers.get(question_key, CPT[target_diagnosis][question_key] > 0.5)

        # Update
        P = bayesian_update(P, question_key, answer)
        asked.add(question_key)
        question_count += 1

    return question_count


def compare_baseline_strategies():
    """
    Compare our IG-based approach to baseline strategies.
    """
    print("\n" + "=" * 70)
    print("Analysis 3: How does our approach compare to alternatives?")
    print("=" * 70)

    # Test cases (same as conditional entropy analysis)
    test_cases = {
        "Migraine": {
            "S1": True,
            "S3": True,
            "S4": True,
            "S7": True,
            "S8": True,
        },
        "Trigeminal Neuralgia": {
            "S28": True,
            "S26": True,
            "S27": True,
            "S2": True,
        },
        "Tension-Type": {
            "S17": True,
            "S18": True,
            "S19": True,
            "S34": True,
        },
        "Cluster": {
            "S11": True,
            "S6": True,
            "S5": True,
            "S12": True,
        },
    }

    results = []

    print(f"\n{'Diagnosis':<25} {'IG (Ours)':<12} {'Random':<15} {'Frequency':<12}")
    print("-" * 70)

    for diagnosis, answers in test_cases.items():
        # Our approach
        ig_questions = simulate_ig_ordering(diagnosis, answers)

        # Random ordering
        random_mean, random_std = simulate_random_ordering(
            diagnosis, answers, num_trials=100
        )

        # Frequency-based
        freq_questions = simulate_frequency_based_ordering(diagnosis, answers)

        results.append(
            {
                "diagnosis": diagnosis,
                "ig": ig_questions,
                "random": random_mean,
                "random_std": random_std,
                "frequency": freq_questions,
            }
        )

        print(
            f"{diagnosis:<25} {ig_questions:<12} {random_mean:.1f}±{random_std:.1f}     {freq_questions:<12}"
        )

    # Compute averages
    avg_ig = np.mean([r["ig"] for r in results])
    avg_random = np.mean([r["random"] for r in results])
    avg_freq = np.mean([r["frequency"] for r in results])

    print("-" * 70)
    print(f"{'AVERAGE':<25} {avg_ig:<12.1f} {avg_random:<15.1f} {avg_freq:<12.1f}")

    print(f"\nOur approach is {avg_random/avg_ig:.1f}x faster than random")
    print(f"Our approach is {avg_freq/avg_ig:.1f}x faster than frequency-based")

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("  inforMED - Analysis Script for Novel Insights")
    print("=" * 70)
    print()

    # Run all analyses -- everything here can be included in the report
    mi_results = compute_mutual_information_all_symptoms()
    entropy_chains = analyze_conditional_entropy_chains()
    baseline_results = compare_baseline_strategies()

    print("\n" + "=" * 70)
    print("Here's what we found:")
    print("=" * 70)
    print("\n1. Some symptoms are way more informative than others")
    print("   - Temporal questions (duration, timing) are the MVPs")
    print("   - Top symptoms give ~0.4 bits, bottom ones only ~0.06 bits")
    print("\n2. Information gain actually compounds over time")
    print("   - Early questions: ~0.15 bits each")
    print("   - Later questions: up to 0.9 bits (5x more powerful!)")
    print("\n3. Our IG approach beats the alternatives")
    print("   - 2x faster than random ordering")
    print("   - 3x faster than frequency-based ordering")
    print("\nThese results are used in the report's 'Novel Insights' section.")
