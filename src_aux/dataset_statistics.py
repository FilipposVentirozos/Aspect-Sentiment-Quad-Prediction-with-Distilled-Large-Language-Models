import os
import ast

def calculate_dataset_statistics(datasets):
    """Calculates and prints statistics for a list of datasets.

    For each dataset, this function calculates the number of sentences, the number of sentences
    with no quadruples, the number of null aspects, the number of null opinions, and the total
    number of quadruples.

    Args:
        datasets (list): A list of dataset names.
    """
    data_path = os.path.join(os.path.dirname(os.getcwd()), "data")
    raw_data_path = os.path.join(data_path, "raw")

    for dataset in datasets:
        data = list()
        with open(os.path.join(raw_data_path, dataset, "test.txt"), 'r') as t:
            lines = t.readlines()
        for line in lines:
            sentence, quadruples_str = line.strip().split('####')
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

if __name__ == "__main__":
    datasets = ["rest15", "rest16", "coursera", "amazon_FF", "hotels"]
    calculate_dataset_statistics(datasets)