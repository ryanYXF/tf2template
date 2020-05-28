# encoding: utf-8
# env: py3.7, tf2.0
# author: ryan.Y

#  data pipeline with tf.data and generator 
#  major concerns are multi threads and multi stage pipeline with low latency 
## reference https://medium.com/@nimatajbakhsh/building-multi-threaded-custom-data-pipelines-for-tensorflow-f76e9b1a32f5
# 

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
        dataset = self._build_dataset_index()
        dataset = self._combine_datasets(dataset, self.generator_file2sample)
        dataset = self._transform_dataset(dataset)
        dataset = self._group_dataset(dataset)
        return dataset
    def _build_dataset_index(self):
        datasetIndex = tf.data.Dataset.range(self.config.dataset_config.gen_num_parallel)
        return datasetIndex
    def _combine_datasets(self, dataset, gen):
        gen_num_parallel = self.config.dataset_config.gen_num_parallel
        datasetCombined = dataset.interleave(lambda element: tf.data.Dataset.from_generator(gen, 
                                                              output_types=(tf.float32), args=(element,)
                                                                                           ),
                                                              cycle_length=gen_num_parallel,
                                                              block_length=1,
                                                              num_parallel_calls=gen_num_parallel
                                            )
        return datasetCombined
    def _transform_dataset(self,dataset):
        """
        operations change individual sample

        Arguments:
            dataset {[tf.data.dataset]} -- [dataset pipeline, samples are tf.tensor]

        Returns:
            a new dataset
        """
        return dataset


    def _group_dataset(self, dataset):
        """
        operations on batch and sequence order, not modify any sample 

        Arguments:
            dataset {[tf.data.dataset]} -- [dataset pipeline, samples are tf.tensor]

        Returns:
            a new dataset
        """
        
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


    
