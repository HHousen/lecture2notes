import logging

import pkg_resources
from symspellpy.symspellpy import SymSpell
from tqdm import tqdm


class SpellChecker:
    """A spell checker."""

    def __init__(
        self,
        max_edit_distance_dictionary=2,
        max_edit_distance_lookup=2,
        prefix_length=7,
    ):
        self.logger = logging.getLogger(__name__)

        # create object
        sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
        # load dictionary
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
        )
        # term_index is the column of the term and count_index is the column of the term frequency
        if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
            self.logger.error("Dictionary file not found")
            return
        if not sym_spell.load_bigram_dictionary(
            bigram_path, term_index=0, count_index=2
        ):
            self.logger.error("Bigram dictionary file not found")
            return

        self.max_edit_distance_lookup = max_edit_distance_lookup
        self.prefix_length = prefix_length
        self.sym_spell = sym_spell

    def check(self, input_term):
        """Checks an input string for spelling mistakes

        Args:
            input_term (str): the sequence to check for spelling errors

        Returns:
            [str]: the best corrected string
        """
        # lookup suggestions for multi-word input strings (supports compound splitting & merging)
        # max edit distance per lookup is now per single word, not per whole input string
        suggestions = self.sym_spell.lookup_compound(
            input_term, self.max_edit_distance_lookup
        )
        # display suggestion term, edit distance, and term frequency
        # for suggestion in suggestions:
        #     print("{}, {}, {}".format(suggestion.term, suggestion.distance,
        #                             suggestion.count))
        output_term_list = [suggestion.term for suggestion in suggestions]

        return output_term_list[0]

    def check_all(self, input_terms):
        """Spell check multiple sequences by calling :meth:`~lecture2notes.end_to_end.spell_check.check` for each item in ``input_terms``.

        Args:
            input_terms (list): a list of strings to be corrected with spell checking

        Returns:
            [list]: a list of corrected strings
        """
        output_terms = []
        for term in tqdm(input_terms, desc="Spell Checking"):
            checked_term = self.check(term)
            output_terms.append(checked_term)
        return output_terms


# Testing
# spell_checker = SpellChecker()
# test_check_result = spell_checker.check("A big long stintg that is bound to have some erros because I am typing it fast and trying to not make mistakes but I probably ma. Shoot I uust hit backspace but that was an error roght thaere so that is good.")
# print(test_check_result)
