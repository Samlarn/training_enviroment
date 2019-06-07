"""
 Just some bad testsetup. 
"""

import DataPipeline.imagePipline as ip


local_path = "C:\\Users\\jonss\\Desktop\\ml\\workspaces\\data\\leafs"

all_image_paths, all_image_labels, image_count = ip.prepare_images(local_path, 192, 192)

print(image_count)
print(all_image_paths[:5])
print(all_image_labels)

zipDataset = ip.get_zip_dataset(all_image_paths, all_image_labels)
print(zipDataset)


mapDataset = ip.get_map_dataset(all_image_paths, all_image_labels)
print(mapDataset)
