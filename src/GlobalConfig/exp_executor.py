"""
author: ouyangtianxiong
date: 2019/12/25
des: implement some experiment executors
"""
class BasicExecutor:
    """
    the base class of all executors
    """
    def __init__(self, net_name, dataset_name, hyper_parameters):
        """
        to execute a series experiment, model, data, and parameters is necessary
        :param net_name: the model training
        :param dataset_name:  the dataset used
        :param hyper_parameters:  hyper_parameters of training
        """
        self.net_name = net_name
        self.dataset_name = dataset_name
        self.hyper_parameters = hyper_parameters
    @TODO
    def execute_one_session(self, session, mode):
        pass

    def execute(self):
        for mode in ['subject_dependent', 'subject_independent']:
            for i in range(1, 4):
                self.execute_one_session(session=i, mode=mode)