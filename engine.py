"""
Engine Module: Information Theory and Bayesian Calculations

This module contains all the mathematical functions for:
- Information theory (entropy, information gain)
- Bayesian inference (probability updates)
- Question selection algorithm
"""

import numpy as np
from data import DIAGNOSES, SYMPTOMS, CPT, NUM_DIAGNOSES

# ============================================================================
# Information Theory Functions
# ============================================================================

def log2(p):
    """
    Returns log base 2 of p, handling p=0.

    By definition in information theory: 0 * log(0) = 0

    Args:
        p (float): Probability value

    Returns:
        float: log2(p) or 0 if p=0
    """
    return np.log2(p) if p > 0 else 0


def calculate_entropy(P):
    """
    Calculates the entropy H(S) for a given probability distribution P.

    Entropy measures uncertainty/information content in bits:
    H(S) = -Σ P(s_i) * log2(P(s_i))

    Args:
        P (list or array): Probability distribution (must sum to 1)

    Returns:
        float: Entropy in bits
    """
    P = np.array(P)
    H = -np.sum(P * np.array([log2(p) for p in P]))
    return H


# ============================================================================
# Bayesian Inference Functions
# ============================================================================

def bayesian_update(current_P, symptom_key, answer_is_yes):
    """
    Updates the probability distribution P using Bayes' Rule and the CPT.

    Applies Bayes' Theorem:
    P(D_k | Answer) = P(Answer | D_k) * P(D_k) / P(Answer)

    Args:
        current_P (list): Current probability distribution over diagnoses
        symptom_key (str): Symptom identifier (e.g., 'S1')
        answer_is_yes (bool): True if patient answered "Yes", False for "No"

    Returns:
        list: Updated (posterior) probability distribution
    """
    new_P = []

    for diagnosis in DIAGNOSES:
        # Get the likelihood P(Answer | D_k) from CPT
        P_S_given_D = CPT[diagnosis][symptom_key]

        # If answer is "No", use complement probability
        if not answer_is_yes:
            P_S_given_D = 1.0 - P_S_given_D

        # Apply Bayes' Rule numerator: P(Answer | D_k) * P(D_k)
        numerator = P_S_given_D * current_P[DIAGNOSES.index(diagnosis)]
        new_P.append(numerator)

    # Normalize the new probabilities (divide by P(Answer))
    denominator = sum(new_P)

    if denominator == 0:
        # If denominator is 0 (shouldn't happen with good CPT), reset to uniform
        return [1.0/NUM_DIAGNOSES] * NUM_DIAGNOSES

    normalized_P = [p / denominator for p in new_P]
    return normalized_P


# ============================================================================
# Information Gain Calculation
# ============================================================================

def calculate_information_gain(current_P, symptom_key):
    """
    Calculates the Information Gain (IG) for asking a specific symptom question.

    IG measures expected reduction in uncertainty:
    IG = H(current) - E[H(new)]

    where E[H(new)] = P(Yes)*H(P|Yes) + P(No)*H(P|No)

    Args:
        current_P (list): Current probability distribution
        symptom_key (str): Symptom identifier to evaluate

    Returns:
        float: Information gain in bits
    """
    # 1. Calculate Current Entropy
    H_current = calculate_entropy(current_P)

    # 2. Calculate P(Yes) and P(No) using Law of Total Probability
    # P(Yes) = Σ P(Yes | D_k) * P(D_k)
    P_yes = sum(CPT[d][symptom_key] * current_P[i]
                for i, d in enumerate(DIAGNOSES))
    P_no = 1.0 - P_yes

    # 3. Calculate Hypothetical Entropies after each possible answer
    P_D_given_yes = bayesian_update(current_P, symptom_key, answer_is_yes=True)
    H_yes = calculate_entropy(P_D_given_yes)

    P_D_given_no = bayesian_update(current_P, symptom_key, answer_is_yes=False)
    H_no = calculate_entropy(P_D_given_no)

    # 4. Calculate Expected New Entropy E[H_new]
    E_H_new = (P_yes * H_yes) + (P_no * H_no)

    # 5. Calculate Information Gain
    IG = H_current - E_H_new

    return IG


# ============================================================================
# Question Selection Algorithm
# ============================================================================

def get_next_question(current_P, asked_symptoms):
    """
    Finds the unasked question with the maximum Information Gain.

    This is the greedy selection algorithm that chooses the question
    expected to reduce uncertainty the most.

    Args:
        current_P (list): Current probability distribution
        asked_symptoms (set): Set of already asked symptom keys

    Returns:
        tuple: (question_text, symptom_key, ig_value) or (None, None, 0) if no questions left
    """
    max_ig = -1
    best_question_key = None

    # Identify questions not yet asked
    unasked_keys = [k for k in SYMPTOMS.keys() if k not in asked_symptoms]

    if not unasked_keys:
        return None, None, 0

    # Evaluate Information Gain for each unasked question
    for symptom_key in unasked_keys:
        ig = calculate_information_gain(current_P, symptom_key)

        if ig > max_ig:
            max_ig = ig
            best_question_key = symptom_key

    # Return the human-readable question, its key, and the IG value
    return SYMPTOMS[best_question_key], best_question_key, max_ig
