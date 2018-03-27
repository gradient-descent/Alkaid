import codecs

from utils import read_lines, SEPARATOR


def get_sentences(sentence_tag, line_num):
    words = []
    for item in sentence_tag.split(' '):
        try:
            index = item.rindex('/')
            words.append(item[:index])
        except ValueError:
            # TODO:Write to log
            print('Line {line_num} occurring \'{item}\' get \'ValueError: substring not found\''
                  .format(line_num=line_num, item=item))
    return ' '.join(words)


def extract_sentences(input_segment_file, sentences_file):
    lines = read_lines(input_segment_file)
    with codecs.open(sentences_file, 'w', encoding='utf-8') as file_w:
        line_num = 1
        for line in lines:
            label_text = line.split(SEPARATOR)
            if len(label_text) < 2:
                label_text.append('')
            word_tag = label_text[1]
            file_w.write('%s\n' % get_sentences(word_tag, line_num))
            line_num += 1
