"""Select the upstream full-catalog filenames before setup.sh builds the index."""
import argparse
from pathlib import Path


def configure(root):
    path = Path(root) / 'web_agent_site/utils.py'
    original = path.read_text()
    text = original
    for key, filename in [('DEFAULT_FILE_PATH','items_shuffle'),('DEFAULT_ATTR_PATH','items_ins_v2')]:
        small = f"{key} = join(BASE_DIR, '../data/{filename}_1000.json')"
        full = f"{key} = join(BASE_DIR, '../data/{filename}.json')"
        if full in text.splitlines():
            continue
        if small not in text.splitlines():
            raise ValueError(f'Unexpected upstream {key}. Check the pinned WebShop revision.')
        text = '\n'.join(full if line == small else line for line in text.split('\n'))
    if original != text:
        path.write_text(text)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('upstream', type=Path)
    configure(p.parse_args().upstream)
