import os.path

import numpy as np
import h5py
from PIL import Image

# type: 0 -pred ; 1 - true/false

colordict = {'Unknown': [255, 255, 255],
                     'Apples': [128, 0, 0],
                     'Beets': [238, 232, 170],
                     'Berries': [255, 107, 70],
                     'Biodiversity area': [0, 191, 255],
                     'Buckwheat': [135, 206, 235],
                     'Chestnut': [0, 0, 128],
                     'Chicory': [138, 43, 226],
                     'Einkorn wheat': [255, 105, 180],
                     'Fallow': [0, 255, 255],
                     'Field bean': [210, 105, 30],
                     'Forest': [65, 105, 225],
                     'Gardens': [255, 140, 0],
                     'Grain': [139, 0, 139],
                     'Hedge': [95, 158, 160],
                     'Hemp': [128, 128, 128],
                     'Hops': [147, 112, 219],
                     'Legumes': [85, 107, 47],
                     'Linen': [176, 196, 222],
                     'Lupine': [127, 255, 212],
                     'Maize': [100, 149, 237],
                     'Meadow': [240, 128, 128],
                     'Mixed crop': [255, 99, 71],
                     'Multiple': [220, 220, 220],
                     'Mustard': [0, 128, 128],
                     'Oat': [0, 206, 209],
                     'Pasture': [106, 90, 205],
                     'Pears': [205, 92, 92],
                     'Peas': [186, 85, 211],
                     'Potatoes': [189, 183, 107],
                     'Pumpkin': [34, 139, 34],
                     'Rye': [184, 134, 11],
                     'Sorghum': [0, 100, 0],
                     'Soy': [199, 21, 133],
                     'Spelt': [25, 25, 112],
                     'Stone fruit': [0, 0, 0],
                     'Sugar beet': [152, 251, 152],
                     'Summer barley': [245, 222, 179],
                     'Summer rapeseed': [32, 178, 170],
                     'Summer wheat': [255, 69, 0],
                     'Sunflowers': [0, 0, 255],
                     'Tobacco': [220, 20, 60],
                     'Tree crop': [255, 255, 102],
                     'Vegetables': [255, 20, 147],
                     'Vines': [0, 20, 200],
                     'Wheat': [255, 215, 0],
                     'Winter barley': [128, 128, 0],
                     'Winter rapeseed': [154, 205, 50],
                     'Winter wheat': [124, 252, 0]}


def visualization(path,img_path,type="predict"):
    print("To path ", path)
    testset = path
    testset = h5py.File(testset, "r")
    if type == "predict":
        # Open hdf5 file
        testset_target = testset["pred"][...]

        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                  26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51]
        label_names = ['Meadow', 'Winter wheat', 'Maize', 'Pasture',
                                 'Sugar beet', 'Winter barley', 'Winter rapeseed',
                                 'Vegetables', 'Potatoes', 'Wheat', 'Sunflowers', 'Vines', 'Spelt']

        # Visualize ground-truth map for the test region
        Mx = 5064 // 24 - 1
        My = 4815 // 24 - 1

        num_patches = Mx * My
        test_shape = [41790,24,24]

        # Generate GT label map
        target_map_image = np.zeros([int(Mx * test_shape[1]), int(My * test_shape[2])])
        step = test_shape[1]
        patch_size = test_shape[1]
        count = 0
        for i_y in range(0, My):
            for i_x in range(0, Mx):
                target_map_image[int(step * i_x):int(step * i_x) + patch_size, int(step * i_y):int(step * i_y) + patch_size] = testset_target[count]
                count += 1

                if count == 8357:
                    break
            if count == 8357:
                break

        target_map_image = target_map_image[:, :step*i_y]
        target_map_RGB = np.ones([target_map_image.shape[0], target_map_image.shape[1], 3]) * 255

        # Map color code from label
        for i_x in range(target_map_RGB.shape[0]):
            for i_y in range(target_map_RGB.shape[1]):
                target_pix_val = target_map_image[i_x, i_y]

                if target_pix_val == 99:
                    continue


                target_pix_color = colordict[label_names[labels.index(target_pix_val)]]
                target_map_RGB[i_x, i_y, :] = np.array(target_pix_color)


        img = Image.fromarray(np.uint8(target_map_RGB))
        img.save(os.path.join(img_path, 'pred_classes.png'))
    else:
        testset_target = testset["tf"][...]

        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                  26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51]
        label_names = ['Meadow', 'Winter wheat', 'Maize', 'Pasture',
                       'Sugar beet', 'Winter barley', 'Winter rapeseed',
                       'Vegetables', 'Potatoes', 'Wheat', 'Sunflowers', 'Vines', 'Spelt']


        # Visulize ground-truth map for the test region
        Mx = 5064 // 24 - 1
        My = 4815 // 24 - 1

        num_patches = Mx * My
        test_shape = [41790, 24, 24]

        # Generate GT label map
        target_map_image = np.zeros([int(Mx * test_shape[1]), int(My * test_shape[2])])
        step = test_shape[1]
        patch_size = test_shape[1]
        count = 0
        for i_y in range(0, My):
            for i_x in range(0, Mx):
                target_map_image[int(step * i_x):int(step * i_x) + patch_size,
                int(step * i_y):int(step * i_y) + patch_size] = testset_target[count]
                count += 1

                if count == 8357:
                    break
            if count == 8357:
                break

        target_map_image = target_map_image[:, :step * i_y]
        target_map_RGB = np.ones([target_map_image.shape[0], target_map_image.shape[1], 3]) * 255

        # Map color code from label
        for i_x in range(target_map_RGB.shape[0]):
            for i_y in range(target_map_RGB.shape[1]):
                target_pix_val = target_map_image[i_x, i_y]

                if target_pix_val == 99:
                    continue
                if target_pix_val == 100:
                    target_pix_color = [0, 255, 0]
                else:
                    target_pix_color = [255, 0, 0]
                target_map_RGB[i_x, i_y, :] = np.array(target_pix_color)

        img = Image.fromarray(np.uint8(target_map_RGB))
        img.save(os.path.join(img_path, 'pred_tf.png'))

    # Close hdf5 file
    testset.close()


if __name__ == "__main__":
    path = f"../../checkpoints_simple/"
    
    img_type = "tf"  # "predict" or "tf"
    h5_path = os.path.join(path, f"last_{img_type}.h5")
    img_path = path
    visualization(h5_path, img_path=img_path, type=img_type)

    img_type = "predict"  # "predict" or "tf"
    h5_path = os.path.join(path, f"last_{img_type}.h5")
    img_path = path
    visualization(h5_path, img_path=img_path, type=img_type)