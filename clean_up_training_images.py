import os
import sys

basenames = sys.argv[1:]


index_range = []

for basename in basenames:
    for filename in os.listdir('images'):
        filepath = os.path.join('images', filename)

        if filename.__contains__(basename):
            number_string = filename[len(basename)+1:-4]
            try:
                index_range.append(int(number_string))
            except ValueError:
                continue

    if len(index_range) < 10:
        print('Aborting due to small number of files.')
    else:
        index_range = sorted(index_range)[:-3]

        for index in index_range:
            os.remove(os.path.join('images', f'{basename}_{index}.png'))

