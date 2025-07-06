import os
import time
from datetime import datetime

research_path = 'memory/layer1_research_docs'
current_time = time.time()

if os.path.exists(research_path):
    files = [f for f in os.listdir(research_path) if f.startswith('research_') and f.endswith('.md')]
    print(f'Total research files: {len(files)}')
    
    recent_files = []
    for f in files[:10]:  # Check first 10 files
        fp = os.path.join(research_path, f)
        ft = os.path.getmtime(fp)
        age_hours = (current_time - ft) / 3600
        age_str = datetime.fromtimestamp(ft).strftime('%Y-%m-%d %H:%M:%S')
        print(f'{f}: {age_hours:.1f} hours old ({age_str})')
        if age_hours < 1:
            recent_files.append(f)
    
    print(f'\nRecent files (< 1 hour): {len(recent_files)}')
    if recent_files:
        for rf in recent_files:
            print(f'  - {rf}')
else:
    print(f'Research path does not exist: {research_path}')