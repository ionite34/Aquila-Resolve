# This reads a CMUDict formatted dictionary as a dictionary object
from .data import DATA_PATH

_default = 'cmudict.dict'


def read_dict(filename: str) -> list:
    # Read the file
    with open(filename, encoding='utf-8', mode='r') as f:
        # Read the file into lines
        lines = f.readlines()
    # Remove any line starting with ";;;"
    lines = [line for line in lines if not line.startswith(';;;')]
    return lines


class DictReader:
    def __init__(self, filename=None):
        self.filename = filename
        self.dict = {}
        # If filename is None, use the default dictionary
        # default = 'data' uses the dictionary file in the data module
        # default = 'nltk' uses the nltk cmudict
        if filename is not None:
            self.dict = self.parse_from_file(filename)
        else:
            with DATA_PATH.joinpath(_default) as f:
                self.dict = self.parse_from_file(f)

    def parse_from_file(self, filename: str) -> dict:
        return self.parse_dict(read_dict(filename))

    def parse_dict(self, lines: list) -> dict:
        # Create a dictionary to store the parsed data
        parsed_dict = {}
        # Detect file format

        # We will read the first 10 lines to determine the format
        # Default to SSD format unless we find otherwise
        dict_form = 'SSD'
        for line in lines[:10]:
            # Strip new lines
            line = line.strip()
            if line == '':
                continue
            """
            Format 1 (Double Space Delimited):
            - Comment allowed to start with ";;;"
            WORD  W ER1 D

            Format 2 (Single Space Delimited):
            - Comment allowed at end of any line using "#"
            WORD W ER1 D # Comment

            Format 3 (Tab Delimited):
            - Detects tab in any line
            """
            if '  ' in line:
                dict_form = 'DSD'
                break

            if '\t' in line:
                dict_form = 'TD'
                break

        # Iterate over the lines
        for line in lines:
            # Skip empty lines
            line = line.strip()
            if not line:
                continue

            # Split depending on format
            if dict_form == 'DSD':
                pairs = line.split('  ')
            elif dict_form == 'TD':
                pairs = line.split('\t')
            elif dict_form == 'SSD':
                space_index = line.find(' ')
                line_split = line[:space_index], line[space_index + 1:]
                pairs = line_split[0], line_split[1].split('#')[0]
            else:
                raise ValueError('Unknown dictionary format')

            word = str.lower(pairs[0])  # Get word and lowercase it
            phonemes = str(pairs[1])  # Get phonemes
            parsed_dict[word] = phonemes  # Store in dictionary

        return parsed_dict
