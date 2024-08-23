import os
import sys

basename = 'musdb_stem1' # sys.argv[1]

print(basename)

index_range = []

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
    sys.exit(0)

index_range = sorted(index_range)[:-3]

for index in index_range:
    os.remove(os.path.join('images', f'{basename}_{index}.png'))

