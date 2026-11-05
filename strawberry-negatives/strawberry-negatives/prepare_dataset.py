"""Create mixed YOLO manifests without modifying the original dataset."""
from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', default='D:/스마트팜/dataset_yolo')
    args = parser.parse_args()
    original = Path(args.original).resolve()
    root = Path(__file__).resolve().parent
    summary = {}
    contents = {}
    seen = set()
    for split in ('train', 'val', 'test'):
        images = sorted(p for p in (original / 'images' / split).iterdir()
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
        if not images:
            raise ValueError(f'No original images: {split}')
        for p in images:
            label = original / 'labels' / split / (p.stem + '.txt')
            if not label.exists():
                raise ValueError(f'Missing original label: {label}')
            for line in label.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5 or fields[0] not in ('0', '1', '2'):
                    raise ValueError(f'Invalid detection label: {label}')
                coords = [float(x) for x in fields[1:]]
                if not all(0 <= x <= 1 for x in coords):
                    raise ValueError(f'Invalid coordinates: {label}')
        negative = sorted((root / 'images' / split).glob('*.jpg'))
        for p in negative:
            label = root / 'labels' / split / (p.stem + '.txt')
            if not label.exists() or label.stat().st_size:
                raise ValueError(f'Expected empty label: {label}')
            if p.stem in seen:
                raise ValueError('Duplicate negative image ID across splits')
            seen.add(p.stem)
        contents[split] = '\n'.join(p.as_posix() for p in images + negative) + '\n'
        summary[split] = {'original': len(images), 'negative': len(negative), 'total': len(images) + len(negative)}
    for split, text in contents.items():
        (root / f'{split}.txt').write_text(text, encoding='utf-8')
    lines = [f'path: {json.dumps(root.as_posix(), ensure_ascii=False)}']
    lines += [f'{split}: {split}.txt' for split in ('train', 'val', 'test')]
    lines += ['names:', '  0: Fullripe', '  1: Semiripe', '  2: Unripe']
    (root / 'data_mixed.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (root / 'dataset_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('READY:', root / 'data_mixed.yaml')


if __name__ == '__main__':
    main()
