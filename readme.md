19-3-17
(1) all small dataset's code and config

19-3-21
(2) two DL applications

19-3-23
(1) alexnet last layer weight decay multiplied by 250
(2) tkde-resnet-param-interval & tkde-alexnet-param-interval: enlarge the param intervals so that the BO searches hyper-parameters in a large intervals. running in slave2 and slave3

19-3-17 - 19-3-23
(1) DL applications (range is 1/10 to 10 of the best weight decay), where alexnet last layer is not multiplied by 250, alexnet results are bad
(2) alexnet last layer is multiplied by 250
(3) tkde-resnet-param-interval & tkde-alexnet-param-interval: enlarge the param intervals (around 1/10000 to 10000 of the best weight decay) so that the BO searches hyper-parameters in a large intervals. running in slave2 and slave3
(4) running 48 hours ...
