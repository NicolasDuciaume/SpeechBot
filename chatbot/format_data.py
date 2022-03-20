# tag = intents file name
# patterns = after the first line
# responses = what's after <Response>

import os
import re
import json

# assign directory
directory = "intents"


def clean(text):
    text.strip()
    text.lower()
    text = re.sub(',;:!', '', text)
    text.replace('?', '')
    return remove_contractions(text)


def remove_contractions(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# iterate over files in the intents directory
list_of_intents = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    # Store the filename as the tag in the intents json
    tag = filename.split(".")[0]
    print(tag)

    intent_dict = {}
    patterns = []
    responses = []
    # iterate through the file to store the patterns and responses
    with open(f) as data_file:
        file_text = data_file.read()
        try:
            p, r = file_text.split(tag)[1].split("<Response>")

            for line in p.strip().splitlines():
                if "------" not in line:
                    str_line = clean(line)
                    patterns.append(str_line)

            for line in r.strip().splitlines():
                str_line = clean(line)

                responses.append(str_line)

        except ValueError as err:
            print(err)
            print("Skipping the file " + filename)
            continue

        intent_dict["tag"] = tag.lower()
        intent_dict["patterns"] = patterns
        intent_dict["responses"] = responses
        list_of_intents.append(intent_dict)

    with open('data.json', 'w+') as fp:
        json.dump(list_of_intents, fp)
