from transformers import BertTokenizer, BertModel
import torch
import regex as re


def process_sentence(sentence, target_term, max_length=150):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    input_ids = tokenizer.encode(sentence, add_special_tokens=True, max_length=max_length, truncation=True)

    input_tensor = torch.tensor([input_ids])

    with torch.no_grad():
        outputs = model(input_tensor)

    hidden_states = outputs.last_hidden_state

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    target_tokens = tokenizer.tokenize(target_term)
    target_length = len(target_tokens)

    token_index = {}

    modified_sentence = []
    target_token_indices = []

    with open('data/text files/token_embeddings.txt', 'w') as f:
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token in ['[CLS]', '[SEP]']:
                i += 1
                continue

            if tokens[i:i + target_length] == target_tokens:
                modified_sentence.append("$T$")
                target_token_indices = []

                for j in range(target_length):
                    word_embedding = hidden_states[0, i + j, :].numpy()
                    embedding_str = ' '.join(map(str, word_embedding))
                    if target_tokens[j] in token_index:
                        token_index[target_tokens[j]] += 1
                    else:
                        token_index[target_tokens[j]] = 0
                    indexed_token = f"{target_tokens[j]}_{token_index[target_tokens[j]]}"
                    f.write(f"{indexed_token} {embedding_str}\n")
                    target_token_indices.append(indexed_token)

                i += target_length
            else:
                if token in token_index:
                    token_index[token] += 1
                else:
                    token_index[token] = 0

                indexed_token = f"{token}_{token_index[token]}"
                modified_sentence.append(indexed_token)

                word_embedding = hidden_states[0, i, :].numpy()
                embedding_str = ' '.join(map(str, word_embedding))
                f.write(f"{indexed_token} {embedding_str}\n")

                i += 1

    modified_sentence_str = ' '.join(modified_sentence)

    with open('data/text files/modified_sentence.txt', 'w') as f:
        f.write(modified_sentence_str + '\n')
        if target_token_indices:
            f.write(' '.join(target_token_indices) + '\n')


def find_target_indices(sentence, target):
    words = sentence.split()

    sentence_str = ' '.join(words)

    match = re.search(re.escape(target), sentence_str)
    if match:
        start_index = match.start()
        end_index = match.end()
        return start_index, end_index
    else:
        return None, None


