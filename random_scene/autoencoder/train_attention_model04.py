from __future__ import print_function
from model04 import *

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    common_args = ["--model-path", "attention", "--epochs", "2"]
    logging_args = ["--log-file", "train_atten.log"]
    main(common_args + ["--attention", "e1", "4"] + logging_args)
    main(common_args + ["--epoch-start", "3", "--attention", "e2", "2"])
    main(common_args + ["--epoch-start", "5", "--attention", "e3", "1"])
    main(common_args + ["--epoch-start", "7", "--attention", "e4", "1"])
    main(common_args + ["--epoch-start", "9", "--attention", "e5", "1"])
    main(common_args + ["--epoch-start", "11", "--attention", "d1", "1"])
    main(common_args + ["--epoch-start", "13", "--attention", "d2", "1"])
    main(common_args + ["--epoch-start", "15", "--attention", "d3", "1"])
    main(common_args + ["--epoch-start", "17", "--attention", "d4", "2"])
    main(common_args + ["--epoch-start", "19", "--attention", "d5", "4"])
