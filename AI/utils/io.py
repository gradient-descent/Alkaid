import codecs


def read_lines(path):
    lines = []
    with codecs.open(path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            line = line.rstrip()
            if line:
                lines.append(line)
    return lines
