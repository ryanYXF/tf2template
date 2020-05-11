# encoding: utf-8
# python3.7, tf2.0
## reference https://medium.com/@nimatajbakhsh/building-multi-threaded-custom-data-pipelines-for-tensorflow-f76e9b1a32f5
import tensorflow as tf

class DatasetFactory():
    """ build dataset from genenator with multi threads and multi stage to reduce the latency of data preparation 
        datasetFactory = DatasetFactory(config)
        dataset = datasetFactory.build_dataset()
    Arguments:
        config object
    
    """
    def __init__(self, config=None):
        ##super().__init__(self, )
        self.config = config
        self.build_dataset()
    def build_dataset(self):
        dataset = self.__build_dataset_index()
        dataset = self.__combine_datasets(dataset, self.generator_file2sample)
        dataset = self.__group__dataset(dataset)
        return dataset
    def __build_dataset_index(self):
        datasetIndex = tf.data.Dataset.range(self.config.dataset_config.gen_num_parallel)
        return datasetIndex
    def __combine_datasets(self, dataset, gen):
        gen_num_parallel = self.config.dataset_config.gen_num_parallel
        datasetCombined = dataset.interleave(lambda element: tf.data.Dataset.from_generator(gen, 
                                                              output_types=(tf.float32), args=(element,)
                                                                                           ),
                                                              cycle_length=gen_num_parallel,
                                                              block_length=1,
                                                              num_parallel_calls=gen_num_parallel
                                            )
        return datasetCombined
    def __transform__dataset(self,dataset):
        pass
    def __group__dataset(self, dataset):
        seed = self.config.dataset_config.shuffle_seed
        batch_size = self.config.dataset_config.batch_size
        drop_remainder = self.config.dataset_config.batch_drop_remainder
        shuffle_buffer_size = self.config.dataset_config.shuffle_buffer_size
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size ,seed=seed)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
    def generator_file2sample(self, element):
        count = 1
        while True:
            yield element * 10 + count
            count += 1


    
