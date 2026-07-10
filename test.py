
import os
from collections import Counter
import json

raw_dir = r'D:\rag-assistant\k8s-finetune\data\raw'
files = os.listdir(raw_dir)
sources = []
for f in files:
    filepath = os.path.join(raw_dir, f)
    with open(filepath, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        sources.append(data['source'])

counts = Counter(sources)
print(f'Total documents: {len(files)}')
for source, count in sorted(counts.items()):
    print(f'  {source}: {count}')
