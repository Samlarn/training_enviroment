"""
 Just some bad testsetup. 
"""

import DataPipeline.imagePipline as ip


local_path = "C:\\Users\\jonss\\Desktop\\ml\\workspaces\\data\\leafs"

zip_dataset = ip.setup_and_get_zipDataset(local_path, 192, 192)

print(zip_dataset)