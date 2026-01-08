from glados.utils.spoken_text_converter import SpokenTextConverter


def test_spoken_text_converter_pl_numbers_do_not_raise() -> None:
    stc = SpokenTextConverter(language="pl")
    result = stc._number_to_words(42)
    assert isinstance(result, str)