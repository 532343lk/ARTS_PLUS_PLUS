import xml.etree.ElementTree as ET
import json


def extract_unique_categories(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    word_target_dict = {}
    conflicts = {}

    for review in root.findall('Review'):
        for sentence in review.findall('./sentences/sentence'):
            opinions = sentence.find('Opinions')
            if opinions is not None:
                for opinion in opinions.findall('Opinion'):
                    target = opinion.get('target').lower()
                    category = opinion.get('category')
                    if target != "NULL" and target != "null":
                        category_first_part = category.split('#')[0]
                        if target.lower() in word_target_dict:
                            if category_first_part != word_target_dict[target.lower()]:
                                conflicts[target.lower()] = category_first_part + " and " + word_target_dict[
                                    target.lower()]
                        else:
                            word_target_dict[target.lower()] = category_first_part
    return word_target_dict, conflicts


xml_file = "data/2016/ABSA-16_SB1_Restaurants_Train_Data.xml"

# Extract unique categories and conflicts
word_target_dict, conflicts = extract_unique_categories(xml_file)

categories_targets_dict = {}

for target, category in word_target_dict.items():
    if category not in categories_targets_dict:
        categories_targets_dict[category] = []
    categories_targets_dict[category].append(target)

with open('data/categories_dicts/target_category_dict.json', 'w') as f:
    json.dump(word_target_dict, f)

with open('data/categories_dicts/categories_targets_dict.json', 'w') as f:
    json.dump(categories_targets_dict, f)

