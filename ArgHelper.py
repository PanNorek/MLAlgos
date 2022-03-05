import argparse


def arg_helper() -> list:
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-m", "--model", default='DecisionTree', help="path to Model File, already: [DecisionTree, RandomForest, KNN, SVCModel, XGB]")
    arg_parse.add_argument("-d", "--data", default='data/dataset1.csv', help="path to Data File ")
    arg_parse.add_argument("-o", "--output", default='DEATH_EVENT', help="name of predicted label")
    args = vars(arg_parse.parse_args())
    return args