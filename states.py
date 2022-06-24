from FeatureCloud.app.engine.app import AppState, app_state, Role


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

    def read_data(self, file_anno, file_exp, tax):
        self.df = preprocessing.Preprocessing.read_data(file_anno, file_exp, tax)

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
        contributions = len([el for el in self.gather_data() if el == True])
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
        #self.register_transition('rerun', Role.PARTICIPANT)

    def run(self):
        common_to_drop = self.await_data()

        c.delete_columns(common_to_drop)

        # apply model here
        ml = FLLogisticRegression()
        ml.prepare_data(c.get_dataframe())
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
        print("### NEW MODEL ###")
        ml.build_new_model(True)
        ml.prepare_data(c.get_dataframe())
        print(ml.benchmark(None, None, True))

        ml.refit_model()
        if c.is_contributing():
            self.send_data_to_coordinator(ml.get_params())

        if self.is_coordinator:
            print("jump to run_aggregate")
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
        print(params)
        ml = FLLogisticRegression()
        ml.set_params(params[0], params[1], False)
        ml.prepare_data(c.get_dataframe())
        ml.build_new_model(True)
        ml.refit_model()
        c.iterations += 1
        print("#%d: %s" % (c.iterations, ml.benchmark(None, None, True)))

        if (self.is_coordinator and c.iterations >= c.iteration_limit+1) or (not self.is_coordinator and c.iterations >= c.iteration_limit):
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
        print('## Aggregation')
        print(all_params)
        ml = FLLogisticRegression()
        for a in all_params:
            ml.set_params(a[0][0], a[1][0])
        ml.aggragate_params()
        self.broadcast_data(ml.get_params())

        return 'rerun'