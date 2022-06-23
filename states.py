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

    def read_data(self, file_anno, file_exp):
        self.df = preprocessing.Preprocessing.read_data(file_anno, file_exp)

    def get_dataframe(self):
        return self.df

    def delete_columns(self, to_drop):
        self.df = self.df.drop(columns=to_drop)

    def is_contributing(self):
        counter = Counter(self.df['CLASS'])
        ratio = counter[0] / (counter[0] + counter[1])

        if ratio > 0.85:
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
        self.register_transition('await', Role.BOTH)

    def run(self):
        c.read_data('/mnt/input/anno.csv', '/mnt/input/exp.csv')

        is_contributing = c.is_contributing()

        self.send_data_to_coordinator(is_contributing)

        if self.is_coordinator:
            return 'aggregate_contributions'
        elif not self.is_coordinator and c.is_contributing():
            print("# i contributing")
            return 'send_empty_columns'
        else:
            print("# I'M NOT CONTRIBUTING!")
            return 'await'


@app_state('aggregate_contributions', Role.COORDINATOR)
class AggregateContributinsState(AppState):

    def register(self):
        self.register_transition('send_empty_columns', Role.COORDINATOR)

    def run(self):
        contributions = len([el for el in self.gather_data() if el == True])
        c.contributors = contributions

        return 'send_empty_columns'


@app_state('send_empty_columns', Role.BOTH)
class AggregateState(AppState):

    def register(self):
        self.register_transition('aggregate_columns_to_drop', Role.COORDINATOR)
        self.register_transition('await', Role.PARTICIPANT)

    def run(self):
        empty_columns = c.get_dataframe().columns[(c.get_dataframe() == 0).all()]

        self.send_data_to_coordinator(list(empty_columns))

        if self.is_coordinator:
            return 'aggregate_columns_to_drop'
        else:
            return 'await'


@app_state('aggregate_columns_to_drop', Role.COORDINATOR)
class AggregateState(AppState):

    def register(self):
        self.register_transition('await', Role.COORDINATOR)

    def run(self):
        to_drop = self.await_data(n=c.contributors)

        self.log('received data from')
        self.log(len(to_drop))

        common_to_drop = set.intersection(*map(set, to_drop))

        self.broadcast_data(common_to_drop)

        return 'await'


@app_state('await', Role.BOTH)
class AwaitState(AppState):

    def register(self):
        self.register_transition('send_global_params', Role.COORDINATOR)
        self.register_transition('apply', Role.PARTICIPANT)

    def run(self):
        common_to_drop = self.await_data()

        c.delete_columns(common_to_drop)

        # apply model here
        ml = FLLogisticRegression()
        ml.prepare_data(c.get_dataframe())
        ml.fit()
        print(ml.benchmark())
        print( ml.benchmark(None, None, True) )
        params = ml.get_params()

        self.send_data_to_coordinator(params)

        if self.is_coordinator:
            return 'send_global_params'
        else:
            return 'apply'


@app_state('send_global_params', Role.COORDINATOR)
class SendGlobalParamsState(AppState):

    def register(self):
        self.register_transition('apply', Role.COORDINATOR)

    def run(self):
        all_params = self.await_data(n=c.contributors)
        print("Number of contributors: %i" % c.contributors)
        #print(all_params)
        ml = FLLogisticRegression()
        for a in all_params:
            ml.set_params(a[0][0], a[1][0])
        ml.aggragate_params()
        final_params = ml.get_params()

        # params mean
        #print(final_params)
        self.broadcast_data(final_params)

        return 'apply'


@app_state('apply', Role.BOTH)
class ApplyState(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self):
        final_params = self.await_data()
        #print("#### FINAL DATA #####")
        #print(final_params)
        ml = FLLogisticRegression()
        ml.set_params(final_params[0], final_params[1], False)
        print("### NEW MODEL ###")
        ml.build_new_model()
        ml.prepare_data(c.get_dataframe())
        print( ml.benchmark(None, None, True) )

        return 'terminal'