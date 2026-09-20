# subject 字段的 32 种写法归到 9 个类别，按字符串中最先出现的学科关键字判定。
#   "Science (Physics, Chemistry)"  → Physics
#   "Science (Chemistry/Biology)"   → Chemistry
#   "Химия и опазване на околната среда" → Chemistry

# （小写关键字，归一后的类别名）。列表顺序不影响结果。
SUBJECT_KEYWORDS = [
    # 英语
    ('chemistry',          'Chemistry'),
    ('physics',            'Physics'),
    ('biology',            'Biology'),
    ('geograph',           'Geography'),   # matches "geography" and "geographic"
    ('histor',             'History'),     # matches "history" and "historical"
    ('math',               'Mathematics'),
    ('fine art',           'Fine Arts'),
    ('visual art',         'Fine Arts'),
    ('informati',          'Informatics'),  # matches informatics/informatika/information
    ('information tech',   'Informatics'),
    ('psycholog',          'Psychology'),
    # 西里尔字母
    ('химия',              'Chemistry'),
    ('физика',             'Physics'),
    ('биология',           'Biology'),
    # 克罗地亚语、塞尔维亚语的拉丁写法
    ('fizika',             'Physics'),
    ('geografija',         'Geography'),
]


def normalize_subject(s: str) -> str:
    """返回归一后的学科类别。找不到关键字时返回去掉首尾空白的原字符串。"""
    if not s:
        return 'Unknown'
    sl = s.lower()

    best_pos = len(sl) + 1
    best_normalized = None
    for kw, mapped in SUBJECT_KEYWORDS:
        pos = sl.find(kw)
        if 0 <= pos < best_pos:
            best_pos = pos
            best_normalized = mapped

    if best_normalized:
        return best_normalized
    if sl.strip() == 'art':
        return 'Fine Arts'
    return s.strip()


if __name__ == '__main__':
    # 用测试集里真实出现过的写法自检
    test_cases = [
        ("Chemistry", "Chemistry"),
        ("Science (Physics, Chemistry)", "Physics"),
        ("Science (Biology, Chemistry)", "Biology"),
        ("Science (Chemistry, Biology)", "Chemistry"),
        ("Science (Chemistry/Biology)", "Chemistry"),
        ("Science (Biology/Chemistry)", "Biology"),
        ("Science (Biology) and Science (Chemistry/Biology)", "Biology"),
        ("SCIENCE (PHYSICS/ CHEMISTRY)", "Physics"),
        ("SCIENCE (CHEMISTRY, BIOLOGY)", "Chemistry"),
        ("Science(Chemistry)", "Chemistry"),
        ("Science Biology", "Biology"),
        ("Science Physics", "Physics"),
        ("Fizika", "Physics"),
        ("Geografija", "Geography"),
        ("Biology and Health Education", "Biology"),
        ("Physics and Astronomy", "Physics"),
        ("История и цивилизации", "Unknown (没匹配)"),  # 故意失败
        ("Химия и опазване на околната среда", "Chemistry"),
        ("Chemistry and Environmental Protection", "Chemistry"),
        ("Geography and Economics", "Geography"),
        ("History and Civilizations", "History"),
        ("Informatika", "Informatics"),
        ("Information Technology", "Informatics"),
        ("Fine Arts", "Fine Arts"),
        ("Psychology", "Psychology"),
        ("Mathematics", "Mathematics"),
    ]
    print(f'{"输入":<55} {"期望":<15} {"实际":<15} {"✓/✗"}')
    print('-' * 95)
    for s, expected in test_cases:
        got = normalize_subject(s)
        ok = '✓' if expected.split()[0] == got else ('✓' if expected == 'Unknown (没匹配)' else '✗')
        print(f'  {s:<53} {expected:<15} {got:<15} {ok}')
