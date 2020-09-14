# Self-Supervised Learning for Facial Action Unit Recognition through Temporal Consistency (BMVC2020 Accepted)

This repository contains PyTorch implementation of [Self-Supervised Learning for Facial Action Unit Recognition through Temporal Consistency](https://www.bmvc2020-conference.com/assets/papers/0861.pdf)

![Image of Overview](https://github.com/intelligent-human-perception-laboratory/temporal-consistency/blob/master/img/overview.png)

Proposed parallel encoders network takes a sequence of frames extracted from a
video. The anchor frame is selected at time t, the sibling frame at t + 1, and the following
frames at equal intervals from t +1+k to t +1+Nk. All input frames are fed to ResNet-18
encoders with shared weights, followed by a fully-connected layer to generate 256d
embeddings. L2-norm is applied on output embeddings. We then compute triplet losses for
adjacent frame pairs along with the fixed anchor frame. In each adjacent pair, the preceding
frame is the positive sample and the following frame is the negative sample. Finally, all
triplet losses are added to form the ranking triplet loss.

# Overview Video
[![](http://img.youtube.com/vi/B4AZU4gsK7o/0.jpg)](http://www.youtube.com/watch?v=B4AZU4gsK7o "")
