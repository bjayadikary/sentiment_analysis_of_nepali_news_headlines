from sys import argv

import re
import os

def process_row_from_shabdakosh_file(word):
    # some word has number in it, like: खापा२
    if word[-1] in "०१२३४५६७८९":
        word = word[:-1]
    # some words have suffix indicators, like: खाप्–नु so, removing hyphen and concatenating
    hyphen = "-"
    if hyphen in word:
        word = word.replace(hyphen, "")
    # some root have aliases, i.e. खान्की/खान्गी, return two tokens
    if "/" in word:
        return word.split("/")

    return [word]


# ई is replaced with इ and same for:
# <ऊ,उ>, <व,ब>, <(श,ष),स>, < ू,ु>, < ी,ि>, < ँ,nill>, <रु,रू>
def normalize_word(nep_word):
    # list of characters that need replacing for normalization
    characters_that_needs_replacing = ['ऊ', 'व', 'श', 'ष', 'ू', 'ी', 'ँ', 'ई']

    # list of substituting characters
    # PS: both lists are aligned together i.e. one that needs to be replaced is in same index as one that is used for replacing
    replacing_characters = ['उ', 'ब', 'स', 'स', 'ु', 'ि', '', 'इ']

    # string is immutable; converting string into list so as to later ease in modifying certain index value
    character_list = list(nep_word)
    for i, character in enumerate(character_list):
        if character in characters_that_needs_replacing:
            idx = characters_that_needs_replacing.index(character)
            character_list[i] = replacing_characters[idx]

    return ''.join(character_list)


def extract_words_from_shabdakosh_file(filename, normalize=False):
    with open(filename, "r", encoding='utf8') as fp:
        # words_extended represent list of rows where row might contain root word extended by some number or aliases
        words_extended_list = fp.read().splitlines()
        words_lists = map(process_row_from_shabdakosh_file, words_extended_list)
        words_lists = list(words_lists)

        if normalize:
            normalized_words_list = list(set([normalize_word(word) for word_list in words_lists for word in word_list]))
            return normalized_words_list
        else:
            words_list = list(set([word for word_list in words_lists for word in word_list]))

        return words_list


def extract_suffix_from_suffix_file(filename):
    with open(filename, 'r', encoding='utf8') as fp:
        suffixes_list = fp.read().splitlines()
        normalized_suffixes_list = [normalize_word(suffix) for suffix in suffixes_list]

        return list(set(normalized_suffixes_list))


def arrange_suffix(suffixes_list):
    # Creating a dictionary based on the length of suffix
    suffixes = {}
    for suffix in suffixes_list:
        suffix_length = len(suffix)
        if suffix_length not in suffixes:
            suffixes[suffix_length] = [suffix]
        else:
            if suffix not in suffixes[suffix_length]:
                suffixes[suffix_length] += [suffix]

    return suffixes


def process_stemming_of_word(shabdakosh_words_list, arranged_suffix_dictionary, input_word):
    # normalize each input_word such that it matches specifics in normalized_shabdakosh_words_list
    input_word = normalize_word(input_word)

    # stemming only words that are not an exact match in nepali dictionary
    if input_word in shabdakosh_words_list:
        return input_word

    suffix_len_keys = list(arranged_suffix_dictionary.keys())
    suffix_len_keys.sort(reverse=True)

    for suffix_len in suffix_len_keys:
        if len(input_word) > suffix_len + 1:
            for suffix in arranged_suffix_dictionary[suffix_len]:
                if input_word.endswith(suffix):
                    return input_word[:-suffix_len]

    return input_word


def stem(shabdakosh_words_list, arranged_suffix_dictionary, input_string):
    # appending in list the final result after applying stemming process
    result = []

    # Removal of spaces before & after string, newlines and tabs
    inputs = input_string.strip()
    inputs = inputs.replace('\n', '')
    inputs = inputs.replace('\t', '')

    # Removing punctuations and numbers
    puncts = r"[.()\"#/@;:<>{}'`‘’+=~|!?,%]"
    nepali_num = r"[०-९]+"
    eng_num = r"[0-9]+"

    inputs = re.sub(puncts, '', inputs)
    inputs = re.sub(nepali_num, '', inputs)
    inputs = re.sub(eng_num, '', inputs)

    # Test on this by returning inputs: "७७ जिल्लामै कांग्रेसले सक्यो अधिवेशन, कहाँ कसले जिते सभापति? (सूची)"
    # Also, for instance, ६.७७% or ४,००० will also be removed following the above process gradually.

    # looping over each word(inflected) to get root(stem) of each
    for each_word in inputs.split():
        result.append(process_stemming_of_word(shabdakosh_words_list, arranged_suffix_dictionary, each_word))

    # joining all the stems arriving at result list and returning it
    return ' '.join(result)


def stem_it(raw_string):
    files_for_stemming_path = os.getcwd() + '/../data_preprocessing/files_for_stemming'
    shabdakosh_file = files_for_stemming_path + '/shabdakosh-words.txt'
    suffix_file = files_for_stemming_path + '/suffix.txt'

    normalized_shabdakosh_words_list = extract_words_from_shabdakosh_file(shabdakosh_file, normalize=True)[1:]
    # not taking first row into consideration since empty string ("")

    normalized_suffixes_list = extract_suffix_from_suffix_file(suffix_file)

    arranged_suffix_dictionary = arrange_suffix(normalized_suffixes_list)

    # getting the result after stemming in stemmed_result
    stemmed_result = stem(normalized_shabdakosh_words_list, arranged_suffix_dictionary, raw_string)

    return stemmed_result
    # print(stemmed_result)

# Incase for direct run through terminal
if __name__ == "__main__":
    if len(argv) == 2:
        stem_it(argv[1])