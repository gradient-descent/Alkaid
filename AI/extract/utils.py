import datetime


def recent_years(term=3):
    """

    :param term:
    :return: Recent n years,n is specified by term
    """
    cur_year = datetime.datetime.now().year
    return [cur_year - idx for idx in range(term)]


def sort_by_value(d):
    """
    Sort a dictionary by its values
    :param d:
    :return:
    """
    items = d.items()
    back_items = [[v[1], v[0]] for v in items]
    back_items.sort(reverse=True)
    return [back_items[i] for i in range(0, len(back_items))]
