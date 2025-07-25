import toml
import os
import xmltodict


categories = set()

def extract_categories(text):
    global categories
    for line in text.split('\n'):
        if '####' in line:
            parts = line.split('####')
            if len(parts) > 1:
                quads = eval(parts[1])
                for quad in quads:
                    if len(quad) > 1:
                        categories.add(quad[1])
    return categories

def extract_categories_xml(xml_content):
    dd = xmltodict.parse(xml_content)
    for sent in dd['sentences']['sentence']:
        for cat in sent['aspectCategories']['aspectCategory']:
            categories.add(cat['@category'])


def process(input_file):
    global categories
    with open(input_file, 'r') as file:
        text = file.read()
    if input_file.endswith(".txt"):
        extract_categories(text)
    elif input_file.endswith(".xml"):
        extract_categories_xml(text)


def save_to_toml(filename):
    global categories
    toml_string = toml.dumps({'categories': list(categories)})
    # Reformat the TOML string to have each category on a new line
    categories_formatted = toml_string.replace('categories = [', 'categories = [\n    ')
    categories_formatted = categories_formatted.replace(', ', ',\n    ')
    categories_formatted = categories_formatted.replace(']', '\n]')
    with open(filename, 'w') as f:
        f.write(categories_formatted)


if __name__ == "__main__":
    for dataset in ["laptop_acos"]:
        data_path = os.path.join(os.getcwd(), "data", "raw")
        # dataset = "hotels"
        input_file = os.path.join(data_path, dataset, 'train.txt')  # Replace with your input file path
        process(input_file)

        config_path = os.path.join(os.getcwd(), "config")
        # output_file = os.path.join(config_path, dataset + '_categories.toml')  # Replace with your desired output file path
        output_file = os.path.join(config_path, dataset.split('/')[0] + '_categories.toml')
        save_to_toml(output_file)

    # for dataset in ["amazon_ff/quads", "coursera/quads", "hotels/quads", "phones_eng/quads"]:
    #     data_path = os.path.join(os.path.dirname(os.getcwd()), "data", "raw", "English_ROAST")
    #     # dataset = "hotels"
    #     input_file = os.path.join(data_path, dataset, 'train.txt')  # Replace with your input file path
    #     process(input_file)

    #     config_path = os.path.join(os.path.dirname(os.getcwd()), "config")
    #     # output_file = os.path.join(config_path, dataset + '_categories.toml')  # Replace with your desired output file path
    #     output_file = os.path.join(config_path, dataset.split('/')[0] + '_categories.toml')
    #     save_to_toml(output_file)

