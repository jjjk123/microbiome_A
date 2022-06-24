import os.path

import pandas
from FeatureCloud.app.engine.app import AppState, app_state, Role
import pickle

import preprocessing
from collections import Counter

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.
from machine_learning import FLLogisticRegression


class DataStorage():
    # number of contributors
    contributors = 1
    iterations = 0
    iteration_limit = 2
    columns_to_drop = []

    def read_data(self, file_anno, file_exp, tax):
        self.df = preprocessing.Preprocessing.read_data(file_anno, file_exp, tax)

    def add_dataframe(self, filename: str):
        self.df = pandas.read_csv(filename)
        self.df = self.df.drop(["Unnamed: 0"], axis=1)

    def get_dataframe(self):
        return self.df

    def delete_columns(self, to_drop):
        self.df = self.df.drop(columns=to_drop)

    def is_contributing(self):
        ratio = sum(self.df['CLASS']) / len(self.df['CLASS'])

        if ratio > 0.85 or ratio < 0.15:
            return False
        else:
            return True


c = DataStorage()


@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('read', Role.BOTH)

    def run(self):
        return 'read'


@app_state('read', Role.BOTH)
class ReadState(AppState):

    def register(self):
        self.register_transition('aggregate_contributions', Role.COORDINATOR)
        self.register_transition('send_empty_columns', Role.BOTH)
        self.register_transition('create_local_models', Role.BOTH)

    def run(self):
        c.read_data('/mnt/input/anno.csv', '/mnt/input/exp.csv', '/mnt/input/taxonomy.tsv')

        is_contributing = c.is_contributing()
        self.send_data_to_coordinator(is_contributing)

        if self.is_coordinator:
            return 'aggregate_contributions'
        elif not self.is_coordinator and c.is_contributing():
            print("# i'm contributing")
            return 'send_empty_columns'
        else:
            print("# I'M NOT CONTRIBUTING!")
            return 'create_local_models'


@app_state('aggregate_contributions', Role.COORDINATOR)
class AggregateContributinsState(AppState):

    def register(self):
        self.register_transition('send_empty_columns', Role.COORDINATOR)

    def run(self):
        contributions = len([el for el in self.gather_data() if el is True])
        self.broadcast_data("collected number of contributors")
        c.contributors = contributions
        return 'send_empty_columns'


@app_state('send_empty_columns', Role.BOTH)
class AggregateState(AppState):

    def register(self):
        self.register_transition('aggregate_columns_to_drop', Role.COORDINATOR)
        self.register_transition('create_local_models', Role.PARTICIPANT)

    def run(self):
        empty_columns = c.get_dataframe().columns[(c.get_dataframe() == 0).all()]

        self.await_data(n=1)
        self.send_data_to_coordinator(list(empty_columns))

        if self.is_coordinator:
            return 'aggregate_columns_to_drop'
        else:
            return 'create_local_models'


@app_state('aggregate_columns_to_drop', Role.COORDINATOR)
class AggregateState(AppState):

    def register(self):
        self.register_transition('create_local_models', Role.COORDINATOR)

    def run(self):
        print(c.contributors)
        to_drop = self.await_data(n=c.contributors)

        self.log('received data from')
        if len(to_drop) and type(to_drop[0]) == bool and type(to_drop[1]) == list:
            to_drop = to_drop[1]

        common_to_drop = set.intersection(*map(set, to_drop))
        self.broadcast_data(common_to_drop)

        return 'create_local_models'


@app_state('create_local_models', Role.BOTH)
class AwaitState(AppState):

    def register(self):
        self.register_transition('send_global_params', Role.COORDINATOR)
        self.register_transition('create_federated_model', Role.PARTICIPANT)

    def run(self):
        common_to_drop = self.await_data()

        c.columns_to_drop = common_to_drop
        c.delete_columns(common_to_drop)

        # apply model here
        ml = FLLogisticRegression()
        ml.prepare_data(c.get_dataframe())
        if len(c.get_dataframe()) > 10:
            ml.fit_cv()
        else:
            ml.fit()
        print(ml.benchmark())
        print(ml.benchmark(None, None, True))
        params = ml.get_params()

        self.send_data_to_coordinator(params)

        if self.is_coordinator:
            return 'send_global_params'
        else:
            return 'create_federated_model'


@app_state('send_global_params', Role.COORDINATOR)
class SendGlobalParamsState(AppState):

    def register(self):
        self.register_transition('create_federated_model', Role.COORDINATOR)

    def run(self):
        all_params = self.await_data(n=c.contributors)
        print("Number of contributors: %i" % c.contributors)
        ml = FLLogisticRegression()
        for a in all_params:
            ml.set_params(a[0][0], a[1][0])
        ml.aggragate_params()
        final_params = ml.get_params()
        self.broadcast_data(final_params)

        return 'create_federated_model'


@app_state('create_federated_model', Role.BOTH)
class ApplyState(AppState):

    def register(self):
        # self.register_transition('terminal')
        self.register_transition('rerun_aggregate', role=Role.COORDINATOR)
        self.register_transition('rerun', role=Role.BOTH)

    def run(self):
        final_params = self.await_data(n=1)
        ml = FLLogisticRegression()
        ml.set_params(final_params[0], final_params[1], False)
        ml.build_new_model(True)
        ml.prepare_data(c.get_dataframe())
        print(ml.benchmark(None, None, True))

        ml.refit_model()
        if c.is_contributing():
            self.send_data_to_coordinator(ml.get_params())

        if self.is_coordinator:
            return 'rerun_aggregate'
        else:
            return 'rerun'


@app_state('rerun', Role.BOTH)
class RerunState(AppState):

    def register(self):
        self.register_transition('rerun_aggregate', role=Role.COORDINATOR)
        self.register_transition('rerun', role=Role.BOTH)
        self.register_transition('terminal', role=Role.BOTH)

    def run(self):
        params = self.await_data(n=1)
        ml = FLLogisticRegression()
        ml.set_params(params[0], params[1], False)
        ml.prepare_data(c.get_dataframe())


        print(c.get_dataframe().columns)

        ml.build_new_model(True)
        ml.refit_model()
        c.iterations += 1
        print("#%d: %s" % (c.iterations, ml.benchmark(None, None, True)))

        # evaluation #
        if os.path.exists("/mnt/input/valid_sample_all.csv"):

            print("start evaluation")
            bench1_data = DataStorage()
            #bench1_data.read_data("/mnt/input/valid_sample_anno.csv", "/mnt/input/valid_sample_exp.csv", '/mnt/input/taxonomy.tsv')
            bench1_data.add_dataframe("/mnt/input/valid_sample_all.csv")
            bench1_data.delete_columns(c.columns_to_drop)
            print(bench1_data.get_dataframe().columns)

            ml.prepare_data(bench1_data.get_dataframe())

            #eml = FLLogisticRegression()
            #eml.prepare_data(bench1_data.get_dataframe())
            #eml.set_params(params[0], params[1], False)
            #eml.build_new_model(True)
            print("#EVAL#SAMPLE# %s" % (ml.benchmark(None, None, True)))


            """
                    ml = FLLogisticRegression()
                    ml.set_params(final_params[0], final_params[1], False)
                    ml.build_new_model(True)
                    ml.prepare_data(c.get_dataframe())
                    print(ml.benchmark(None, None, True))
            """

            """
            bench2_data = DataStorage()
            bench2_data.read_data("/mnt/input/valid_country_anno.csv", "/mnt/input/valid_country_exp.csv", '/mnt/input/taxonomy.tsv')
            print(bench2_data.get_dataframe().columns)
            cml = FLLogisticRegression()
            cml.prepare_data(bench2_data.get_dataframe())
            cml.set_params(params[0], params[1], False)
            cml.prepare_data(bench2_data.get_dataframe())
            cml.build_new_model(True)
            print("#EVAL#Country# %s" % (cml.benchmark(None, None, True)))
            """

        if c.iterations >= c.iteration_limit:
            pickle.dump(ml.get_model(), open("/mnt/output/model.p", "wb"))
            return 'terminal'
        else:
            self.send_data_to_coordinator(ml.get_params())
            if self.is_coordinator:
                return 'rerun_aggregate'
            else:
                return 'rerun'


@app_state('rerun_aggregate', Role.COORDINATOR)
class RerunAggregation(AppState):

    def register(self):
        self.register_transition('rerun', role=Role.COORDINATOR)

    def run(self):
        all_params = self.await_data(n=c.contributors)
        #print('## Aggregation')
        #self.log(len(all_params))
        ml = FLLogisticRegression()
        for a in all_params:
            ml.set_params(a[0][0], a[1][0])
        ml.aggragate_params()
        self.broadcast_data(ml.get_params())

        return 'rerun'
