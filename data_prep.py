from data import *
#if you don't want to do data augmentation, set data_gen_args as an empty dict.
#data_gen_args = dict()

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGenerator = trainGenerator(20,'c:/Users/balud/Desktop/folder_ueri/membrane/train','image','label',data_gen_args,save_to_dir = "c:/Users/balud/Desktop/folder_ueri/data/membrane/train/aug")






