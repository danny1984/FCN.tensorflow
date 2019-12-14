"""
AutoPaintBatchDataset: auto paint batch dataset
Author: Danny Gao
"""
import os
import numpy as np
import scipy.misc as misc


class AutoPaintBatchDataset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0
    
    img_files = []
    lbl_files = []

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.image_options = image_options
        self._read_images()
    
    def __init__(self, img_list, lbl_list, image_options={}, IS_TRAIN_VALID="TRAIN"):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.img_files = img_list
        self.lbl_files = lbl_list
        self.image_options = image_options
        #self._read_images()

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(os.path.join(self.image_options["dataset_path_pre"], filename)) for filename in self.img_files])
        self.__channels = False
        self.annotations = np.array(
            [np.expand_dims(self._transform(os.path.join(self.image_options["dataset_path_pre"], filename)), axis=3) for filename in self.lbl_files])
        print (self.images.shape)
        print (self.annotations.shape)

    def _read_images_crt_batch(self, img_files_crt_batch=None, lbl_files_crt_batch=None):
        self.__channels = True
        self.images_crt_batch = np.array([self._transform(os.path.join(self.image_options["dataset_path_pre"], filename)) for filename in img_files_crt_batch])
        self.__channels = False
        self.annotations_crt_batch = np.array(
            [np.expand_dims(self._transform(os.path.join(self.image_options["dataset_path_pre"], filename)), axis=3) for filename in lbl_files_crt_batch])
        print(len(self.images_crt_batch))
        print(len(self.annotations_crt_batch))

    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    # def next_batch(self, batch_size):
    #     start = self.batch_offset
    #     self.batch_offset += batch_size
    #     if self.batch_offset > self.images.shape[0]:
    #         # Finished epoch
    #         self.epochs_completed += 1
    #         print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
    #         # Shuffle the data
    #         perm = np.arange(self.images.shape[0])
    #         np.random.shuffle(perm)
    #         self.images = self.images[perm]
    #         self.annotations = self.annotations[perm]
    #         # Start next epoch
    #         start = 0
    #         self.batch_offset = batch_size
    #
    #     end = self.batch_offset
    #     return self.images[start:end], self.annotations[start:end]

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
#        if self.batch_offset > self.images.shape[0]:
        if self.batch_offset > len(self.img_files):

            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
#            perm = np.arange(self.images.shape[0])
            perm = np.arange(len(self.img_files))

            np.random.shuffle(perm)
            #self.images = self.images[perm]
            self.img_files = self.img_files[perm]
            #self.annotations = self.annotations[perm]
            self.lbl_files = self.lbl_files[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset

        self._read_images_crt_batch(img_files_crt_batch=self.img_files[start:end], lbl_files_crt_batch=self.lbl_files[start:end])

        return self.images_crt_batch, self.annotations_crt_batch

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]