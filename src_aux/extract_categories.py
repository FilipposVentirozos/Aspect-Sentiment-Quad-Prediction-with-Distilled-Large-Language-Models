import toml
import os
import xmltodict

categories = set()

def extract_categories_from_text(text):
    """Extracts categories from a text file.

    Args:
        text (str): The content of the text file.

    Returns:
        set: A set of unique categories.
    """
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

def extract_categories_from_xml(xml_content):
    """Extracts categories from an XML file.

    Args:
        xml_content (str): The content of the XML file.
    """
    dd = xmltodict.parse(xml_content)
    for sent in dd['sentences']['sentence']:
        for cat in sent['aspectCategories']['aspectCategory']:
            categories.add(cat['@category'])

def process_file(input_file):
    """Processes a file to extract categories.

    Args:
        input_file (str): The path to the input file.
    """
    global categories
    with open(input_file, 'r') as file:
        text = file.read()
    if input_file.endswith(".txt"):
        extract_categories_from_text(text)
    elif input_file.endswith(".xml"):
        extract_categories_from_xml(xml_content)

def save_categories_to_toml(filename):
    """Saves the extracted categories to a TOML file.

    Args:
        filename (str): The path to the output TOML file.
    """
    global categories
    toml_string = toml.dumps({'categories': list(categories)})
    categories_formatted = toml_string.replace('categories = [', 'categories = [\n    ')
    categories_formatted = categories_formatted.replace(', ', ',\n    ')
    categories_formatted = categories_formatted.replace(']', '\n]')
    with open(filename, 'w') as f:
        f.write(categories_formatted)

if __name__ == "__main__":
    for dataset in ["laptop_acos"]:
        data_path = os.path.join(os.getcwd(), "data", "raw")
        input_file = os.path.join(data_path, dataset, 'train.txt')
        process_file(input_file)

        config_path = os.path.join(os.getcwd(), "config")
        output_file = os.path.join(config_path, dataset.split('/')[0] + '_categories.toml')
        save_categories_to_toml(output_file)