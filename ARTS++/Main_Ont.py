from OntologyReasoner import OntReasoner

test_path_laptop_og = "/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/HAABSA_input_files/2014Laptop/original/test.txt"
test_path_laptop_ARTS = "/Volumes/LaCie/Econometrie Master/Thesis/Data_Eva_laptop/data/HAABSA_input_files/2014Laptop/ARTS/test.txt"
test_path_laptop_ARTS_PLUS_PLUS = "data/HAABSA_input_files/LaptopARTS++test.txt"

print('Starting Ontology Reasoner')

Ontology = OntReasoner()
accuracyOnt, remaining_size = Ontology.run(True, test_path_laptop_ARTS_PLUS_PLUS, False)
print('test acc={:.4f}, remaining size={}'.format(accuracyOnt, remaining_size))


