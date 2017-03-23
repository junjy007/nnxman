import tensorflow as tf
from experiment_manage.core import Builder, BuilderFactory


class TrainBuilder(Builder):
    def __init__(self, conf):
        super(TrainBuilder, self).__init__(conf)
        self.conf = conf
        self.learning_rate = self.get_learning_rate()
        self.optim = None
        self.train_op = None
        return

    # noinspection
    def get_learning_rate(self):
        """
        Dummy process, assuming the lr_opt is a real number of the learning rate
        :return:
        """
        return self.conf['solver']['learning_rate']

    def get_optimiser(self):
        """
        In future, construct specified optimiser. This needs to be a member function,
        so the build_train can call (then we will have the loss-tensor, and can
        explicitly impose dependencies on the loss).
        :return:
        """
        optim = tf.train.GradientDescentOptimizer(
                learning_rate=self.learning_rate,
                name='GradDescent')
        return optim

    def build(self, inputs, phrase):
        assert phrase == 'train'
        loss = inputs[0]
        with tf.control_dependencies([loss, ]):
            self.optim = self.get_optimiser()
            assert isinstance(self.optim, tf.train.Optimizer)
            grads_and_vars = self.optim.compute_gradients(loss=loss)
        train_op = self.optim.apply_gradients(grads_and_vars)
        return [train_op]


class TrainBuilderFactory(BuilderFactory):
    def __init__(self):
        super(TrainBuilderFactory, self).__init__()

    def get_builder(self, conf):
        return TrainBuilder(conf)


def id_str():
    return "Architecture construction:", __file__
