
with open('kladi/data/mouse_TFs.txt', 'r') as f:
    mouse_tfs = [x.strip() for x in f][1:]

with open('kladi/data/human_TFs.txt', 'r') as f:
    human_tfs = [x.strip() for x in f][1:]