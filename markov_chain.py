import numpy as np

with open('markov-chain/cleaned_output.txt', 'r') as f:
    text = f.read()

def matrix_construction(text):
    matrix = {}
    for i in range(len(text) - 1):
        key = text[i]
        next_char = text[i+1]
        if key not in matrix:
            matrix[key] = []
        matrix[key].append(next_char)
    return matrix

M = np.zeros((27, 27), dtype=int)
char_to_index = {char: idx for idx, char in enumerate('abcdefghijklmnopqrstuvwxyz ')}
index_to_char = {idx: char for char, idx in char_to_index.items()}

for i in range(len(text) - 1):
    current_char = text[i]
    next_char = text[i + 1]
    if current_char in char_to_index and next_char in char_to_index:
        current_index = char_to_index[current_char]
        next_index = char_to_index[next_char]
        M[current_index][next_index] += 1

def stochastic_matrix(M):
    row_sums = M.sum(axis=1)
    stochastic_M = np.zeros_like(M, dtype=float)
    for i in range(len(M)):
        if row_sums[i] > 0:
            stochastic_M[i] = M[i] / row_sums[i]
    return stochastic_M

stochastic_M = stochastic_matrix(M)

with open('markov-chain/test_text.txt', 'r') as f:
    test_text = f.read()

def evaluate_markov_performance(P, test_text, char_idx):
    score = 0
    tentatives = 0
    for i in range(len(test_text) - 1):
        x_char, y_char = test_text[i], test_text[i+1]
        if x_char in char_idx and y_char in char_idx:
            x, y = char_idx[x_char], char_idx[y_char]
            score += P[x, y]
            tentatives += 1
    return score / tentatives

performance_score = evaluate_markov_performance(stochastic_M, test_text, char_to_index)
print(performance_score)
# it goes from 0.12976917843560656 to 0.13940689229211797 since we use qwen to clean the text, which is a significant improvement. This means that the Markov chain model is better at predicting the next character in the cleaned text compared to the previous version

def get_state_index(state, char_idx):
    return char_idx[state[0]] * 729 + char_idx[state[1]] * 27 + char_idx[state[2]]

M_3 = np.zeros((19683, 27), dtype=int)

for i in range(len(text) - 3):
    state = text[i:i+3]
    next_char = text[i+3]
    if all(c in char_to_index for c in state) and next_char in char_to_index:
        state_idx = get_state_index(state, char_to_index)
        next_idx = char_to_index[next_char]
        M_3[state_idx, next_idx] += 1

row_sums_3 = M_3.sum(axis=1)
stochastic_M_3 = np.zeros_like(M_3, dtype=float)

for i in range(19683):
    if row_sums_3[i] > 0:
        stochastic_M_3[i] = M_3[i] / row_sums_3[i]

score_3 = 0
tentatives_3 = 0

for i in range(len(test_text) - 3):
    state = test_text[i:i+3]
    next_char = test_text[i+3]
    if all(c in char_to_index for c in state) and next_char in char_to_index:
        state_idx = get_state_index(state, char_to_index)
        next_idx = char_to_index[next_char]
        score_3 += stochastic_M_3[state_idx, next_idx]
        tentatives_3 += 1

performance_score_3 = score_3 / tentatives_3 if tentatives_3 > 0 else 0
print(performance_score_3)
#0.4261259749447424 is a significant improvement over the previous score of 0.13940689229211797, indicating that the 3-character Markov chain model is much better

def generate_text_3rd_order(stochastic_M_3, char_to_index, index_to_char, length=100):
    valid_states = np.where(stochastic_M_3.sum(axis=1) > 0)[0]
    current_index = np.random.choice(valid_states)
    
    c1 = index_to_char[current_index // 729]
    c2 = index_to_char[(current_index % 729) // 27]
    c3 = index_to_char[current_index % 27]
    current_state = c1 + c2 + c3
    
    result = list(current_state)
    
    for _ in range(length - 3):
        probabilities = stochastic_M_3[current_index]
        if probabilities.sum() == 0:
            break
        
        next_index = np.random.choice(27, p=probabilities)
        next_char = index_to_char[next_index]
        result.append(next_char)
        
        current_state = current_state[1:] + next_char
        current_index = char_to_index[current_state[0]] * 729 + char_to_index[current_state[1]] * 27 + char_to_index[current_state[2]]
        
    return ''.join(result)

generated_text = generate_text_3rd_order(stochastic_M_3, char_to_index, index_to_char, length=15)
print(generated_text)