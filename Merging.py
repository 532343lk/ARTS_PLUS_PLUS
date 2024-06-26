import json


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def construct_item(item_id, item_data, sentence):
    return {
        "from": item_data["from"],
        "id": item_id,
        "polarity": item_data["polarity"],
        "sentence": sentence,
        "term": item_data["term"],
        "to": item_data["to"]
    }


def merge_json_files(original_file, revTgt_file, revNon_file, addDiff_file, output_file):
    # Load all JSON files
    original_data = load_json(original_file)
    revTgt_data = load_json(revTgt_file)
    revNon_data = load_json(revNon_file)
    addDiff_data = load_json(addDiff_file)

    merged_data = {}

    for original_id, original_item in original_data.items():
        sentence = original_item["sentence"]
        for subitem_id, subitem in original_item["term_list"].items():
            # Add original subitem
            merged_data[subitem_id] = construct_item(subitem_id, subitem, sentence)

            # Add corresponding items from revTgt, revNon, and addDiff if they exist
            adv_suffix = 1
            if subitem_id in revTgt_data:
                merged_data[f"{subitem_id}_adv{adv_suffix}"] = construct_item(f"{subitem_id}_adv{adv_suffix}",
                                                                              revTgt_data[subitem_id],
                                                                              revTgt_data[subitem_id]["sentence"])
            adv_suffix += 1
            if subitem_id in revNon_data:
                merged_data[f"{subitem_id}_adv{adv_suffix}"] = construct_item(f"{subitem_id}_adv{adv_suffix}",
                                                                              revNon_data[subitem_id],
                                                                              revNon_data[subitem_id]["sentence"])
            adv_suffix += 1
            if subitem_id in addDiff_data:
                merged_data[f"{subitem_id}_adv{adv_suffix}"] = construct_item(f"{subitem_id}_adv{adv_suffix}",
                                                                              addDiff_data[subitem_id],
                                                                              addDiff_data[subitem_id]["sentence"])

    # Save the merged data to the output file
    save_json(merged_data, output_file)


# Define file paths
original_file = 'data/2014/laptop/test_sent.json'
revTgt_file = 'data/2014/laptop/output/ARTS++/test/revTgt.json'
revNon_file = 'data/2014/laptop/output/ARTS++/test/revNon.json'
addDiff_file = 'data/2014/laptop/output/ARTS++/test/addDiff.json'
output_file = 'data/2014/laptop/output/ARTS++/test/merged.json'

# Merge the files
merge_json_files(original_file, revTgt_file, revNon_file, addDiff_file, output_file)
