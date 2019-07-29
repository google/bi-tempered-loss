/* Copyright 2019 Google Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

function logT(u, t) {
    if (t == 1) {
        return tf.log(u)
    } else {
        return tf.div(tf.sub(tf.pow(u, (1.0 - t)), 1.0), (1.0 - t))
    }
}


function expT(u, t) {
    if (t == 1) {
        return tf.exp(u)
    } else {
        return tf.relu(tf.pow(tf.add(1.0, tf.mul((1.0 - t), u)), (1.0 / (1.0 - t))))
    }

}

function computeNormalization(activations, t, numIters = 5) {
    mu = tf.max(activations, -1, true)
    normalizedActivationsStep0 = tf.sub(activations, mu)

    normalizedActivations = normalizedActivationsStep0
    for (var i = 0; i < numIters; ++i) {
        logtPartition = tf.sum(expT(normalizedActivations, t), -1, true)
        normalizedActivations = tf.mul(normalizedActivationsStep0, tf.pow(
            logtPartition, 1 - t))
    }
    logtPartition = tf.sum(expT(normalizedActivations, t), -1, true)
    return tf.sub(mu, logT(tf.div(1.0, logtPartition), t))
}

function temperedSoftmax(activations, t, numIters = 5) {
    if (t == 1.0) {
        normalizationConstants = tf.log(tf.sum(tf.exp(activations), -1, true))
    } else {
        normalizationConstants = computeNormalization(activations, t, numIters)
    }
    diff = tf.sub(activations, normalizationConstants)
    return expT(diff, t)
}

function temperedSigmoid(activations, t, numIters = 5) {
    activations2d = tf.reshape(activations, [-1, 1])
    internalLogits = tf.concat([tf.zerosLike(activations2d), activations2d], 1)
    return temperedSoftmax(internalLogits, t, numIters)
}

function bitemperedLogisticLoss(activations, labels, t1, t2, numIters = 5) {
    probabilities = temperedSoftmax(activations, t2, numIters)

    lossValues = tf.sub(tf.mul(labels, tf.sub(logT(tf.add(labels, 1e-10), t1), logT(probabilities, t1))),
        tf.mul(tf.div(1.0, (2.0 - t1)), tf.sub(tf.pow(labels, 2.0 - t1), tf.pow(probabilities, 2.0 - t1))))
    return tf.sum(lossValues, -1)
}

function bitemperedBinaryLogisticLoss(activations, labels, t1, t2, numIters = 5) {
    outShape = labels.shape
    labels2d = tf.reshape(labels, [-1, 1])
    activations2d = tf.reshape(activations, [-1, 1])
    labels2d = tf.reshape(labels, [-1, 1])
    zeroLabel2d = tf.sub(1.0, labels2d)
    internalLabels = tf.concat([zeroLabel2d, labels2d], 1)
    internalLogits = tf.concat([tf.zerosLike(activations2d), activations2d], 1)
    losses = bitemperedLogisticLoss(internalLogits, internalLabels, t1, t2, numIters)
    return tf.reshape(losses, outShape)
}