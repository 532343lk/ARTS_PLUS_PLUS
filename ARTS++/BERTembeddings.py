from transformers import BertTokenizer, BertModel
import torch

def process_sentence(sentence, target_term, tokenizer, model, token_global_index, max_length=150):
    input_ids = tokenizer.encode(sentence, add_special_tokens=True, max_length=max_length, truncation=True)

    input_tensor = torch.tensor([input_ids])

    with torch.no_grad():
        outputs = model(input_tensor)

    hidden_states = outputs.last_hidden_state

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    target_tokens = tokenizer.tokenize(target_term)
    target_length = len(target_tokens)

    modified_sentence = []
    target_token_indices = []

    token_embeddings = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in ['[CLS]', '[SEP]']:
            i += 1
            continue

        if tokens[i:i + target_length] == target_tokens:
            for target_token in target_tokens:
                if target_token in token_global_index:
                    token_global_index[target_token] += 1
                else:
                    token_global_index[target_token] = 0
                indexed_target_token = f"{target_token}_{token_global_index[target_token]}"
                target_token_indices.append(indexed_target_token)
                word_embedding = hidden_states[0, i, :].numpy()
                embedding_str = ' '.join(map(str, word_embedding))
                token_embeddings.append((indexed_target_token, embedding_str))
                i += 1

            modified_sentence.append("$T$")
        else:
            if token in token_global_index:
                token_global_index[token] += 1
            else:
                token_global_index[token] = 0

            indexed_token = f"{token}_{token_global_index[token]}"
            modified_sentence.append(indexed_token)

            word_embedding = hidden_states[0, i, :].numpy()
            embedding_str = ' '.join(map(str, word_embedding))
            token_embeddings.append((indexed_token, embedding_str))

            i += 1

    modified_sentence_str = ' '.join(modified_sentence)

    return modified_sentence_str, token_embeddings, target_token_indices

def process_file(input_filename, output_sentence_filename, output_embedding_filename, max_length=150):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    combined_sentences = []
    combined_embeddings = []
    token_global_index = {}

    with open(input_filename, 'r') as f:
        lines = f.read().strip().split('\n')
        i = 0
        while i < len(lines):
            sentence = lines[i].strip()
            target_term = lines[i + 1].strip()
            sentiment = lines[i + 2].strip()

            modified_sentence, token_embeddings, target_token_indices = process_sentence(
                sentence.replace("$T$", target_term), target_term, tokenizer, model, token_global_index, max_length
            )

            combined_sentences.append(f"{modified_sentence}\n{' '.join(target_token_indices)}\n{sentiment}")

            combined_embeddings.extend(token_embeddings)

            i += 3

    with open(output_sentence_filename, 'w') as f:
        f.write('\n'.join(combined_sentences))

    with open(output_embedding_filename, 'w') as f:
        for token, embedding in combined_embeddings:
            f.write(f"{token} {embedding}\n")

# Usage
input_filename = 'data/HAABSA_input_files/LaptopARTS++complete.txt'
output_sentence_filename = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/temporaryData/temp_BERT_Base/Laptop/ARTS++/complete.txt'
output_embedding_filename = '/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/final_embeddings/Laptop/ARTS++/complete.txt'
process_file(input_filename, output_sentence_filename, output_embedding_filename)
