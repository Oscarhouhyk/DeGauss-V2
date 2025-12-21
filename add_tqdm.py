import os

file_path = 'train.py'

with open(file_path, 'r') as f:
    content = f.read()

search_str = """    viewpoint_stack_index = list(range(len(train_cams)))
    if not viewpoint_stack and not optimization_params.dataloader:
        # Manual sampling mode - copy camera list
        viewpoint_stack = [i for i in train_cams]
        viewpoint_stack_index_save = copy.deepcopy(viewpoint_stack_index)"""

replace_str = """    viewpoint_stack_index = list(range(len(train_cams)))
    if not viewpoint_stack and not optimization_params.dataloader:
        # Manual sampling mode - copy camera list
        print("Loading all training cameras into memory (this may take a while)...")
        viewpoint_stack = [i for i in tqdm(train_cams, desc="Loading cameras")]
        viewpoint_stack_index_save = copy.deepcopy(viewpoint_stack_index)"""

if search_str in content:
    content = content.replace(search_str, replace_str)
    print("Applied tqdm to train.py")
else:
    print("Could not find context for tqdm in train.py")

with open(file_path, 'w') as f:
    f.write(content)
