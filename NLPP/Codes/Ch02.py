"""
@Description: A Quick Tour of Traditional NLP
@Author(s): Stephen CUI
@LastEditor(s): somebody name
@CreatedTime: 2023-04-23 11:17:21
"""

from typing import List


def n_grams(text: List[str], n: int) -> List[list]:
    """生成 n-gram

    Args:
        text (list): _description_
        n (int): _description_

    Returns:
        list: _description_
    """
    assert len(text) >= n
    return [text[i:i + n] for i in range(len(text) - n + 1)]


if __name__ == '__main__':
    pass
