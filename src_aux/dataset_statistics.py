import os
import ast
data_path = os.path.join(os.path.dirname(os.getcwd()), "data")

raw_data_path = os.path.join(data_path, "raw") #, "English_ROAST")

datasets = ["rest15", "rest16", "coursera", "amazon_FF", "hotels"]
# datasets = ["amazon_ff/quads", "coursera/quads", "hotels/quads", "phones_eng/quads"]

for dataset in datasets:
    data = list()
    # Count the number of sentences
    with open(os.path.join(raw_data_path, dataset, "test.txt"), 'r') as t:
        lines = t.readlines()
    for line in lines:
        # Split the line by the delimiter '####'
        sentence, quadruples_str = line.strip().split('####')
        # Convert the quadruples string to a list of lists using ast.literal_eval
        quadruples = ast.literal_eval(quadruples_str)

        data.append({
            'sentence': sentence,
            'quadruples': quadruples
        })
    count_void_sentences = 0
    aspect_nulls = 0
    opinions_nulls = 0
    all_quads = 0
    print(dataset)
    for row in data:
        if len(row['quadruples']) == 0:
            # print(row['quadruples'])
            count_void_sentences += 1
        for quadruple in row['quadruples']:
            if quadruple[0] == 'NULL':
                aspect_nulls += 1
            if quadruple[3] == 'NULL':
                opinions_nulls += 1
            all_quads += 1
    print("void sentences:")
    print(count_void_sentences)
    print(count_void_sentences / len(data))

    print("aspect nulls:")
    print(aspect_nulls)
    print("opinion nulls:")
    print(opinions_nulls)
    print("all_quads:")
    print(all_quads)
    print()


