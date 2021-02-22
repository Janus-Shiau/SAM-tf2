import tensorflow as tf


class SAMWarpper:
    """ The wrapper for optimizers using Sharpness-Aware Minimization.

    The mechanism is proposed by P. Foret et.al in the paper
    [Sharpness-Aware Minimization for Efficiently Improving Generalization]

    Example of usage:
    ```
    opt = tf.keras.optimizers.SGD(learning_rate)
    opt = SAMWarpper(opt, rho=0.05)

    def grad_func(inputs, labels):
        with tf.GradientTape() as tape:
            pred = model(inputs, training=True)
            loss = loss_func(pd=pred, gt=label)
        return pred, loss, tape

    opt.optimize(grad_func, model.trainable_variables)
    ```
    """

    def __init__(self, optimizer, rho=0.0, **kwargs):
        """ Wrap optimizer with sharpness-aware minimization.

        Args:
            optimizer: tensorflow optimizer.
            rho: the pertubation hyper-parameter.
        """
        self.optimizer = optimizer
        self.rho = rho
        self.var_list = None

    def optimize(self, grad_func, variables, **kwargs):
        """ API for wrapped optimizer.

        Args:
            grad_func: function return prediction, loss, and gradient tape.
            variables: list of variables to be optimized.

        Returns:
            pred: prediction of the model (defined in grad_func).
            loss: loss value of the original loss function.
            loss_sam: loss value considering SAM.
        """
        if self.rho == 0.0:
            pred, loss, tape = grad_func()
            grads = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
            return pred, loss, 0.0

        if self.var_list is None:
            self._initialize_var(variables)

        pred, loss, tape = grad_func()
        grads = tape.gradient(loss, variables)

        # ? Not sure if the control_dependencies is still required in v2 style
        save_op = [self.var_list[var.name].assign(var) for var in variables]
        with tf.control_dependencies(save_op):
            grads = self.dual_gradients(grads)
            noise_op = [var.assign_add(self.rho * grad) for grad, var in zip(grads, variables)]

        with tf.control_dependencies(noise_op):
            _, loss_sam, tape_sam = grad_func()
            grads = tape_sam.gradient(loss_sam, variables)

        restore_op = [var.assign(self.var_list[var.name]) for var in variables]
        with tf.control_dependencies(restore_op):
            self.optimizer.apply_gradients(zip(grads, variables))

        return pred, loss, loss_sam

    def _initialize_var(self, variables):
        """ Initialized variables for saving addictional varaibles.
        """
        self.var_list = {}
        for var in variables:
            self.var_list[var.name] = tf.Variable(var.value, trainable=False, dtype=var.dtype)

    def dual_gradients(self, grads):
        """ Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.
        """

        grad_norm = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.math.square(grad)) for grad in grads]))
        normalized_grads = [grad / grad_norm for grad in grads]

        return normalized_grads


if __name__ == "__main__":
    import random

    # make simple Dense network
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(16,)))
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # fake dataset: uniform dist. and normal dist.
    data = []
    for i in range(1000):
        data.append(
            {
                "x": tf.random.uniform([1, 16], dtype="float32"),
                "y": tf.ones([1, 1], dtype="float32"),
            }
        )
        data.append(
            {
                "x": tf.random.normal([1, 16], dtype="float32"),
                "y": tf.zeros([1, 1], dtype="float32"),
            }
        )

    # Optimizer setting
    optimizer = tf.keras.optimizers.SGD(0.005)
    optimizer = SAMWarpper(optimizer, rho=0.05)  # rho=0.0 -> disable SAM

    for i in range(10000):
        idx = random.randint(0, len(data) - 1)
        inputs, labels = data[idx]["x"], data[idx]["y"]

        def grad_func():
            with tf.GradientTape() as tape:
                pred = model(inputs)
                loss = tf.reduce_mean(tf.losses.binary_crossentropy(labels, pred))

            return pred, loss, tape

        pred, loss, loss_sam = optimizer.optimize(grad_func, model.trainable_variables)

        if i % 100 == 0:
            print(pred, labels, loss, loss_sam)

