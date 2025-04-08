import streamlit as st
import os
import json
import math
import datetime
from pathlib import Path
from collections import defaultdict

# Create necessary folders
for folder in ['input', 'logs', 'data']:
    Path(folder).mkdir(exist_ok=True)

class LanguageLearningModel:
    """Advanced native implementation of vocabulary tracking and sentence selection"""
    
    def __init__(self):
        # Core vocabulary knowledge structure
        self.vocab = defaultdict(lambda: {
            'language': '',
            'acquisition_score': 0.1,  # 0-1 knowledge level
            'stability': 1.0,          # retention stability in days
            'difficulty': 0.5,         # intrinsic difficulty 0-1
            'last_seen': datetime.datetime.now(),
            'last_forgotten': datetime.datetime.now(),
            'history': [],             # list of past interactions
            'exposures': 0
        })
        
        # Language-specific user parameters
        self.language_ability = defaultdict(float)
        
        # Learning parameters
        self.params = {
            'learning_rate': 0.05,     # general learning rate
            'forgetting_rate': 0.03,   # general forgetting rate
            'optimal_known_ratio': 0.7,  # target for i+1 (70% known words is ideal)
            'optimal_new_ratio': 0.2,   # target for i+1 (20% new words is ideal)
            'retrieval_practice': 0.7,  # bonus for successfully recalled words
            'failure_penalty': 0.3,     # penalty for failed recalls
            'time_decay_factor': 0.8    # controls the steepness of forgetting curve
        }
        
        # Load existing model if it exists
        self.load_model()
    
    def load_model(self):
        """Load vocabulary model from disk using JSON"""
        if os.path.exists('data/vocab_model.json'):
            try:
                with open('data/vocab_model.json', 'r') as f:
                    saved_data = json.load(f)
                    
                    # Convert dates from strings to datetime objects
                    if 'vocab' in saved_data:
                        for word, data in saved_data['vocab'].items():
                            if 'last_seen' in data and isinstance(data['last_seen'], str):
                                try:
                                    data['last_seen'] = datetime.datetime.fromisoformat(data['last_seen'])
                                except:
                                    data['last_seen'] = datetime.datetime.now()
                                    
                            if 'last_forgotten' in data and isinstance(data['last_forgotten'], str):
                                try:
                                    data['last_forgotten'] = datetime.datetime.fromisoformat(data['last_forgotten'])
                                except:
                                    data['last_forgotten'] = datetime.datetime.now()
                                    
                            if 'history' in data:
                                for entry in data['history']:
                                    if 'timestamp' in entry and isinstance(entry['timestamp'], str):
                                        try:
                                            entry['timestamp'] = datetime.datetime.fromisoformat(entry['timestamp'])
                                        except:
                                            entry['timestamp'] = datetime.datetime.now()
                                            
                        # Create defaultdict from the saved regular dict
                        vocab_dict = saved_data['vocab']
                        self.vocab = defaultdict(lambda: {
                            'language': '',
                            'acquisition_score': 0.1,
                            'stability': 1.0,
                            'difficulty': 0.5,
                            'last_seen': datetime.datetime.now(),
                            'last_forgotten': datetime.datetime.now(),
                            'history': [],
                            'exposures': 0
                        })
                        for word, data in vocab_dict.items():
                            self.vocab[word] = data
                    
                    if 'language_ability' in saved_data:
                        self.language_ability = defaultdict(float)
                        for lang, ability in saved_data['language_ability'].items():
                            self.language_ability[lang] = ability
                        
                    if 'params' in saved_data:
                        self.params = saved_data['params']
                        
            except json.JSONDecodeError:
                st.warning("Error loading vocabulary: JSON format error. Starting fresh.")
            except Exception as e:
                st.error(f"Error loading vocabulary: {e}")

    def save_model(self):
        """Save vocabulary model to disk using JSON"""
        try:
            # Convert defaultdict to regular dict
            vocab_dict = {}
            for word, data in self.vocab.items():
                # Deep copy to avoid reference issues
                word_data = dict(data)
                
                # Convert datetime objects to ISO format strings
                if 'last_seen' in word_data and isinstance(word_data['last_seen'], datetime.datetime):
                    word_data['last_seen'] = word_data['last_seen'].isoformat()
                if 'last_forgotten' in word_data and isinstance(word_data['last_forgotten'], datetime.datetime):
                    word_data['last_forgotten'] = word_data['last_forgotten'].isoformat()
                    
                # Convert history datetime objects
                if 'history' in word_data:
                    for entry in word_data['history']:
                        if 'timestamp' in entry and isinstance(entry['timestamp'], datetime.datetime):
                            entry['timestamp'] = entry['timestamp'].isoformat()
                            
                vocab_dict[word] = word_data
                
            # Convert language ability defaultdict to dict
            language_ability_dict = dict(self.language_ability)
            
            # Prepare data for serialization
            save_data = {
                'vocab': vocab_dict,
                'language_ability': language_ability_dict,
                'params': self.params
            }
            
            # Write to file
            with open('data/vocab_model.json', 'w') as f:
                json.dump(save_data, f, indent=2)
                
        except Exception as e:
            st.error(f"Error saving vocabulary model: {e}")
    
    def update_knowledge(self, words, language, feedback_level):
        """
        Update vocabulary knowledge based on user feedback
        
        Parameters:
        - words: list of words in the sentence
        - language: language code
        - feedback_level: 1=didn't understand, 2=partially understood, 3=fully understood
        """
        if not words:
            return
            
        # Convert feedback levels to learning scores
        feedback_map = {
            1: 0.1,  # didn't understand
            2: 0.5,  # partially understood
            3: 0.9   # fully understood
        }
        feedback_score = feedback_map.get(feedback_level, 0.5)
        
        # Apply time-based forgetting to all words before update
        self._apply_forgetting_curves()
        
        # Timestamp for this interaction
        now = datetime.datetime.now()
        
        # Update each word
        words_normalized = [w.lower() for w in words if w.strip()]
        unique_words = set(words_normalized)
        
        # Track words for difficulty normalization
        difficulty_sum = 0
        words_processed = 0
        expected_retrievals = []
        
        for word in unique_words:
            # Skip empty strings
            if not word.strip():
                continue
                
            # Get or create word entry
            word_data = self.vocab[word]
            
            # Set language if not already set
            if not word_data['language']:
                word_data['language'] = language
            
            # Calculate retrieval expectation (what we predicted)
            expected_retrieval = self._calculate_retrieval_probability(word)
            expected_retrievals.append(expected_retrieval)
            
            # Record interaction
            word_data['history'].append({
                'timestamp': now,
                'feedback': feedback_level,
                'context_size': len(words)
            })
            
            # Update exposure count and timestamp
            word_data['exposures'] += 1
            word_data['last_seen'] = now
            word_data['last_forgotten'] = now
            
            # Calculate error between expectation and actual feedback
            error = feedback_score - expected_retrieval
            
            # Update acquisition score based on error
            learning_modifier = self.params['retrieval_practice'] if feedback_score >= 0.5 else self.params['failure_penalty']
            word_data['acquisition_score'] += self.params['learning_rate'] * error * learning_modifier
            
            # Constrain acquisition score to valid range
            word_data['acquisition_score'] = max(0.01, min(0.99, word_data['acquisition_score']))
            
            # Update stability based on feedback
            if feedback_score > 0.5:  # Successful recall strengthens memory
                # More successful recalls lead to higher stability increases
                word_data['stability'] *= 1.25
                word_data['stability'] = min(100.0, word_data['stability'])  # Cap at 100 days
            elif feedback_score == 0.5:
                word_data['stability'] *= 1.1
                word_data['stability'] = min(100.0, word_data['stability'])  # Cap at 100 days
            else:
                # Failed recalls decrease stability
                word_data['stability'] *= 0.7  # Partial credit for partial understanding
            
            # Accumulate difficulty for normalization
            difficulty_sum += word_data['difficulty']
            words_processed += 1
        
        # Update language ability based on overall feedback
        prior_ability = self.language_ability.get(language, 0)
        ability_change = self.params['learning_rate'] * (feedback_score - 0.5)  # Positive for good feedback, negative for bad
        if ability_change > 0:
            ability_change *= (1.0 + (words_processed / 20.0))  # Give more credit for longer interactions
        new_ability = prior_ability + ability_change
        self.language_ability[language] = new_ability
        
        # Recalculate word difficulties to better track intrinsic difficulty vs. learner ability
        if words_processed > 0:
            avg_difficulty = difficulty_sum / words_processed
            
            # Adjust difficulty toward average for all words in this interaction
            for word, expected_retrieval in zip(unique_words, expected_retrievals):
                if word.strip():
                    # Words more easily understood than expected should be easier
                    if feedback_score > expected_retrieval:
                        # Move difficulty down (easier) toward the average
                        self.vocab[word]['difficulty'] = (
                            self.vocab[word]['difficulty'] * 0.9 + avg_difficulty * 0.1
                        )
                    else:
                        # Move difficulty up (harder) away from average
                        self.vocab[word]['difficulty'] = (
                            self.vocab[word]['difficulty'] * 0.9 + 
                            min(1.0, self.vocab[word]['difficulty'] * 1.1) * 0.1
                        )
        
        # Save updated model
        self.save_model()
    
    def calculate_sentence_difficulty(self, sentence_unit, target_language):
        """
        Calculate the difficulty score of a sentence using advanced i+1 metrics
        
        Returns a score where higher values are better candidates for i+1 learning
        """
        if not sentence_unit.get("words") or len(sentence_unit["words"]) == 0:
            return 0
            
        words = [w.lower() for w in sentence_unit["words"] if w.strip()]
        unique_words = set(words)
        
        # Skip empty sentences
        if not unique_words:
            return 0
            
        # Get word knowledge metrics
        known_count = 0
        partial_count = 0
        unknown_count = 0
        word_retrieval_probs = []
        
        # Apply forgetting curves to ensure current knowledge state
        self._apply_forgetting_curves()
        
        # Calculate metrics for each word
        for word in unique_words:
            prob = self._calculate_retrieval_probability(word)
            word_retrieval_probs.append(prob)
            
            if prob > 0.8:
                known_count += 1
            elif prob > 0.3:
                partial_count += 1
            else:
                unknown_count += 1
        
        # Calculate key ratios
        total_words = len(unique_words)
        known_ratio = known_count / total_words if total_words > 0 else 0
        partial_ratio = partial_count / total_words if total_words > 0 else 0
        unknown_ratio = unknown_count / total_words if total_words > 0 else 1
        
        # Calculate i+1 score based on optimal learning zone
        
        # 1. Proximity to ideal known ratio (we want about 70% known words)
        known_proximity = 1.0 - abs(known_ratio - self.params['optimal_known_ratio'])
        
        # 2. Proximity to ideal new word ratio (we want about 20% new words)
        new_proximity = 1.0 - abs(unknown_ratio - self.params['optimal_new_ratio'])
        
        # 3. Sentence length factor (preference for shorter sentences when starting out)
        ability = self.language_ability.get(target_language, 0)
        length_factor = 1.0
        if ability < 0.5:  # For beginners, shorter is better
            length_factor = 1.0 - (0.5 * (total_words / 20))  # Normalize by 20 words
            length_factor = max(0.5, length_factor)  # Don't penalize too much
        
        # 4. Word frequency bonus (if we have statistics on word usage)
        # This would prioritize common words over rare ones
        # Not implemented in this version but could be added
        
        # 5. Learning value (sentences with partially known words have more learning value)
        learning_value = partial_ratio * 1.5  # Boost sentences with partially known words
        
        # Combine factors with weights
        i_plus_one_score = (
            (known_proximity * 0.4) +  # 40% weight on known words ratio
            (new_proximity * 0.3) +    # 30% weight on new words ratio
            (length_factor * 0.1) +    # 10% weight on sentence length
            (learning_value * 0.2)     # 20% weight on learning value
        )
        
        # Scale to 0-100
        return i_plus_one_score * 100
    
    def find_optimal_sentences(self, aligned_text, current_indices, target_language, 
                              num_forced=3, min_natural_score=60):
        """
        Find optimal sentences for learning based on current knowledge state
        
        Parameters:
        - aligned_text: list of sentence units with source/target text and words
        - current_indices: indices already selected
        - target_language: language to learn
        - num_forced: number of top sentences to force-select
        - min_natural_score: minimum score for natural candidates
        
        Returns:
        - forced_sentences: list of indices for force-selected sentences
        - natural_sentences: list of indices for naturally qualified sentences
        """
        available_indices = [i for i in range(len(aligned_text)) if i not in current_indices]
        
        # No sentences available
        if not available_indices:
            return [], []
            
        # Calculate scores for all available sentences
        sentence_scores = []
        for i in available_indices:
            score = self.calculate_sentence_difficulty(aligned_text[i], target_language)
            sentence_scores.append((i, score))
        
        # Sort by score (highest first)
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(sentence_scores)
        
        # Select forced sentences (top N)
        forced = [x[0] for x in sentence_scores[:min(num_forced, len(sentence_scores))]]
        
        # Select natural candidates (remaining sentences above threshold)
        natural = [x[0] for x in sentence_scores[min(num_forced, len(sentence_scores)):] 
                  if x[1] >= min_natural_score]
        
        return forced, natural
    
    def get_natural_candidates(self, aligned_text, current_indices, target_language, min_score=60):
        """Find naturally qualifying sentences based on current knowledge"""
        available_indices = [i for i in range(len(aligned_text)) if i not in current_indices]
        natural = []
        
        for i in available_indices:
            score = self.calculate_sentence_difficulty(aligned_text[i], target_language)
            if score >= min_score:
                natural.append(i)
                
        return natural
    
    def _calculate_retrieval_probability(self, word):
        """
        Calculate probability of successful word retrieval based on:
        - Current acquisition level
        - Time-based memory decay
        - Word difficulty
        - User language ability
        """
        # If word not in vocabulary, return baseline probability
        if word not in self.vocab:
            return 0.1
            
        word_data = self.vocab[word]
        
        # If never seen or no language assigned, return baseline
        if word_data['exposures'] == 0 or not word_data['language']:
            return 0.1
            
        # Get current acquisition level adjusted for time decay
        days_since = (datetime.datetime.now() - word_data['last_seen']).total_seconds() / (24*3600)
        
        # Apply forgetting curve using stability parameter
        # Higher stability means slower decay
        decay_factor = 2 ** (-days_since / word_data['stability'])
        
        # Current memory strength
        memory_strength = word_data['acquisition_score'] * decay_factor
        
        # Adjust for word difficulty vs. language ability
        language = word_data['language']
        ability_advantage = self.language_ability.get(language, 0) - word_data['difficulty']
        
        # Apply sigmoid transformation to ability advantage
        ability_factor = 1.0 / (1.0 + math.exp(-ability_advantage))
        
        # Combine memory strength with ability factor
        final_prob = (0.8 * memory_strength) + (0.2 * ability_factor)
        
        # Keep within bounds
        return max(0.01, min(0.99, final_prob))
    
    def _apply_forgetting_curves(self):
        """Apply forgetting curves to all words based on time since last seen"""
        now = datetime.datetime.now()
        
        for word, data in self.vocab.items():
            if data['exposures'] > 0:
                days_since = math.floor((now - data['last_forgotten']).total_seconds() / (24*3600))
                if days_since > 0:
                    # Apply forgetting curve - more stable memories decay slower
                    decay_factor = 2 ** (-days_since / data['stability'] * self.params['time_decay_factor'])
                    data['acquisition_score'] *= decay_factor
                    # Ensure minimum value
                    data['acquisition_score'] = max(0.01, data['acquisition_score'])
                    data['last_forgotten'] = now
    
    def get_vocabulary_statistics(self, target_language=None):
        """Get statistics about vocabulary knowledge"""
        self._apply_forgetting_curves()  # Ensure current knowledge state
        
        stats = {
            'total_words': 0,
            'well_known': 0,  # acquisition > 0.8
            'familiar': 0,    # acquisition 0.4-0.8
            'learning': 0,    # acquisition 0.1-0.4
            'by_language': {},
            'language_abilities': dict(self.language_ability),
            'average_acquisition': 0,
            'average_stability': 0
        }
        
        acquisition_sum = 0
        stability_sum = 0
        
        for _, data in self.vocab.items():
            # Skip words with no exposures
            if data['exposures'] == 0:
                continue
                
            # Filter by target language if specified
            if target_language and data['language'] != target_language:
                continue
                
            language = data['language']
            acquisition = data['acquisition_score']
            
            # Initialize language entry if not exists
            if language not in stats['by_language']:
                stats['by_language'][language] = {
                    'total': 0,
                    'well_known': 0,
                    'familiar': 0,
                    'learning': 0
                }
            
            # Update global counters
            stats['total_words'] += 1
            acquisition_sum += acquisition
            stability_sum += data['stability']
            
            # Update knowledge level counters
            if acquisition > 0.8:
                stats['well_known'] += 1
                stats['by_language'][language]['well_known'] += 1
            elif acquisition > 0.4:
                stats['familiar'] += 1
                stats['by_language'][language]['familiar'] += 1
            else:
                stats['learning'] += 1
                stats['by_language'][language]['learning'] += 1
                
            # Update language total
            stats['by_language'][language]['total'] += 1
            
        # Calculate averages
        if stats['total_words'] > 0:
            stats['average_acquisition'] = acquisition_sum / stats['total_words']
            stats['average_stability'] = stability_sum / stats['total_words']
            
        return stats
    
    def get_word_details(self, word, include_history=False):
        """Get detailed information about a specific word"""
        word = word.lower()
        if word not in self.vocab or self.vocab[word]['exposures'] == 0:
            return None
            
        # Apply forgetting to ensure current state
        self._apply_forgetting_curves()
        
        word_data = self.vocab[word]
        details = {
            'word': word,
            'language': word_data['language'],
            'acquisition_score': word_data['acquisition_score'],
            'stability_days': word_data['stability'],
            'difficulty': word_data['difficulty'],
            'exposures': word_data['exposures'],
            'last_seen': word_data['last_seen'],
            'retrieval_probability': self._calculate_retrieval_probability(word)
        }
        
        if include_history:
            details['history'] = word_data['history']
            
        return details
        
    def reset_vocabulary(self):
        """Reset the vocabulary model"""
        self.vocab = defaultdict(lambda: {
            'language': '',
            'acquisition_score': 0.1,
            'stability': 1.0,
            'difficulty': 0.5,
            'last_seen': datetime.datetime.now(),
            'history': [],
            'exposures': 0
        })
        self.language_ability = defaultdict(float)
        self.save_model()


def rerun_app():
    st.rerun()

def save_session_state(file_path, revealed_indices, target_indices, visible_states, reviewed_indices):
    """Save the current session state to a log file."""
    data = {
        "revealed": list(revealed_indices),
        "target_indices": list(target_indices),
        "visible_states": {str(k): v for k, v in visible_states.items()},
        "reviewed": list(reviewed_indices)
    }
    with open(file_path, 'w') as f:
        json.dump(data, f)

def main():
    st.title("i+1 Language Learning Tool")
    
    # Initialize language model
    if 'language_model' not in st.session_state:
        st.session_state.language_model = LanguageLearningModel()
    model = st.session_state.language_model
    
    # File selection
    input_files = [f for f in os.listdir("input") if f.endswith('.json')]
    if not input_files:
        st.warning("No input files found. Please add JSON files to the 'input' folder.")
        return

    if 'current_file' not in st.session_state:
        st.session_state.current_file = None

    selected_file = st.selectbox("Select a file to read:", input_files)

    # Reset session state when file changes
    if st.session_state.current_file != selected_file:
        for key in list(st.session_state.keys()):
            if key not in ['current_file', 'language_model']:
                del st.session_state[key]
        st.session_state.current_file = selected_file

    target_language = st.selectbox("Target language:", ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Turkish"])

    # Load selected file contents
    file_path = os.path.join("input", selected_file)
    if "aligned_text" not in st.session_state:
        with open(file_path, 'r', encoding='utf-8') as f:
            st.session_state.aligned_text = json.load(f)

    total_units = len(st.session_state.aligned_text)

    # Load logs for this file
    log_file = os.path.join("logs", f"{selected_file}.json")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
                if isinstance(log_data, dict):
                    revealed_indices = set(log_data.get("revealed", []))
                    visible_states = {int(k): v for k, v in log_data.get("visible_states", {}).items()}
                    reviewed_indices = set(log_data.get("reviewed", []))
                    target_indices = log_data.get("target_indices", log_data.get("revealed", []))
                else:
                    revealed_indices = set(log_data)
                    visible_states = {i: True for i in revealed_indices}
                    target_indices = list(revealed_indices)
                    reviewed_indices = set()
            except:
                revealed_indices = set()
                visible_states = {}
                target_indices = []
                reviewed_indices = set()
    else:
        revealed_indices = set()
        visible_states = {}
        target_indices = []
        reviewed_indices = set()

    # Find naturally qualifying sentences
    natural_candidates = model.get_natural_candidates(
        st.session_state.aligned_text, target_indices, target_language
    )

    # Initialize to_replace_indices with stored target indices plus natural candidates
    if 'to_replace_indices' not in st.session_state:
        combined_indices = set(target_indices + natural_candidates)
        if len(combined_indices) < 10:
            forced_needed = 10 - len(combined_indices)
            forced_indices, _ = model.find_optimal_sentences(
                st.session_state.aligned_text, list(combined_indices), target_language, forced_needed
            )
            combined_indices.update(forced_indices)
        st.session_state.to_replace_indices = list(combined_indices)

    remaining_sentences = total_units - len(st.session_state.to_replace_indices)
    potential_natural = len(model.get_natural_candidates(
        st.session_state.aligned_text, st.session_state.to_replace_indices, target_language
    ))

    # Vocabulary statistics in sidebar
    with st.sidebar:
        st.subheader("Vocabulary Statistics")
        
        vocab_stats = model.get_vocabulary_statistics(target_language)
        
        # Display general stats
        st.write(f"Total vocabulary: {vocab_stats['total_words']} words")
        st.write(f"Well known: {vocab_stats['well_known']} words")
        st.write(f"Familiar: {vocab_stats['familiar']} words")
        st.write(f"Still learning: {vocab_stats['learning']} words")
        
        if target_language in vocab_stats['language_abilities']:
            ability = vocab_stats['language_abilities'][target_language]
            st.write(f"Language ability: {ability:.2f}")
        
        # Reset vocabulary button
        if st.button("Reset Vocabulary"):
            if st.session_state.get('confirm_reset', False):
                model.reset_vocabulary()
                st.success("Vocabulary reset!")
                st.session_state.confirm_reset = False
                rerun_app()
            else:
                st.session_state.confirm_reset = True
                st.warning("Click again to confirm reset. This will delete all vocabulary data!")
        
        # Cancel reset
        if st.session_state.get('confirm_reset', False):
            if st.button("Cancel Reset"):
                st.session_state.confirm_reset = False
                rerun_app()

    # Number of new sentences to add
    new_replacements = st.number_input(
        "Force new target sentences:",
        min_value=0,
        max_value=remaining_sentences,
        value=min(3, remaining_sentences),
        step=1,
        help="Number of additional sentences to force into target language"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add Target Sentences"):
            forced_indices, natural_indices = model.find_optimal_sentences(
                st.session_state.aligned_text,
                st.session_state.to_replace_indices,
                target_language,
                new_replacements
            )
            new_indices = forced_indices + natural_indices
            st.session_state.to_replace_indices.extend(new_indices)
            save_session_state(log_file, revealed_indices, st.session_state.to_replace_indices, visible_states, reviewed_indices)
            st.success(f"Added {len(forced_indices)} forced and {len(natural_indices)} natural target sentences!")
            rerun_app()
    with col2:
        transformed_count = len(st.session_state.to_replace_indices)
        revealed_count = sum(1 for idx in st.session_state.to_replace_indices if idx in revealed_indices)
        st.info(
            f"Transformed: {transformed_count}/{total_units} | "
            f"Revealed: {revealed_count}/{transformed_count} | "
            f"New: {potential_natural}"
        )

    st.subheader("Reading Text")
    st.markdown("""
    <style>
        .reviewed {
            color: #28a745;
            font-weight: bold;
        }
        .unrevealed {
            color: #007bff;
        }
        .revealed {
            color: #fd7e14;
            font-style: italic;
        }
    </style>
    """, unsafe_allow_html=True)

    for i, unit in enumerate(st.session_state.aligned_text):
        if i in st.session_state.to_replace_indices:
            is_revealed = i in revealed_indices
            is_reviewed = i in reviewed_indices
            is_source_visible = visible_states.get(i, True) if is_revealed else False

            if is_reviewed:
                prefix = "✓ "
                css_class = "reviewed"
            elif is_revealed:
                prefix = "👁️ "
                css_class = "revealed"
            else:
                prefix = "🔍 "
                css_class = "unrevealed"

            if not is_revealed:
                if st.button(f"{prefix}{unit['target']}", key=f"sentence_{i}"):
                    revealed_indices.add(i)
                    visible_states[i] = True
                    save_session_state(log_file, revealed_indices, st.session_state.to_replace_indices, visible_states, reviewed_indices)
                    feedback_key = f"feedback_{i}"
                    st.session_state[feedback_key] = True
                    rerun_app()
            else:
                if is_source_visible:
                    st.markdown(f"<div class='{css_class}'>{prefix}{unit['target']}</div>", unsafe_allow_html=True)
                    st.info(unit['source'])
                    if st.button("Hide source", key=f"hide_{i}"):
                        visible_states[i] = False
                        save_session_state(log_file, revealed_indices, st.session_state.to_replace_indices, visible_states, reviewed_indices)
                        rerun_app()
                else:
                    if st.button(f"{prefix}{unit['target']}", key=f"show_{i}"):
                        visible_states[i] = True
                        save_session_state(log_file, revealed_indices, st.session_state.to_replace_indices, visible_states, reviewed_indices)
                        rerun_app()
                feedback_key = f"feedback_{i}"
                if is_source_visible and not is_reviewed and st.session_state.get(feedback_key, False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Didn't understand", key=f"feedback_1_{i}"):
                            model.update_knowledge(unit.get('words', []), target_language, 1)
                            st.session_state[feedback_key] = False
                            reviewed_indices.add(i)
                            save_session_state(log_file, revealed_indices, st.session_state.to_replace_indices, visible_states, reviewed_indices)
                            rerun_app()
                    with col2:
                        if st.button("Partially understood", key=f"feedback_2_{i}"):
                            model.update_knowledge(unit.get('words', []), target_language, 2)
                            st.session_state[feedback_key] = False
                            reviewed_indices.add(i)
                            save_session_state(log_file, revealed_indices, st.session_state.to_replace_indices, visible_states, reviewed_indices)
                            rerun_app()
                    with col3:
                        if st.button("Fully understood", key=f"feedback_3_{i}"):
                            model.update_knowledge(unit.get('words', []), target_language, 3)
                            st.session_state[feedback_key] = False
                            reviewed_indices.add(i)
                            save_session_state(log_file, revealed_indices, st.session_state.to_replace_indices, visible_states, reviewed_indices)
                            rerun_app()
        else:
            st.write(unit['source'])

    if st.button("I understood all other target sentences"):
        newly_marked = 0
        for i in st.session_state.to_replace_indices:
            if i not in revealed_indices:
                model.update_knowledge(st.session_state.aligned_text[i].get('words', []), target_language, 3)
                revealed_indices.add(i)
                reviewed_indices.add(i)
                visible_states[i] = False
                newly_marked += 1
        save_session_state(log_file, revealed_indices, st.session_state.to_replace_indices, visible_states, reviewed_indices)
        st.success(f"Marked {newly_marked} remaining sentences as understood!")
        rerun_app()

if __name__ == "__main__":
    main()