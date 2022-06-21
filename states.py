from FeatureCloud.app.engine.app import AppState, app_state, Role
import pandas as pd
from preprocessing import analyse_data
from logistic_regression import logistic_regression
 

# FeatureCloud requires that apps define the at least the 'initial' state.
# This state is executed after the app instance is started.

class Data():
    def read_data(self, file_anno, file_exp):
        # df = pd.DataFrame()
        anno = pd.read_csv(file_anno, delimiter=',')
        exp = pd.read_csv(file_exp, delimiter=',')

        df = analyse_data(anno, exp)

        self.df = df
    
    def get_dataframe(self):
        return self.df
    
    def delete_columns(self, to_drop):
        self.df = self.df.drop(columns=to_drop)

c = Data()

@app_state('initial')
class InitialState(AppState):

    def register(self):
        self.register_transition('read', Role.BOTH)
        
    def run(self):
        return 'read'

@app_state('read', Role.BOTH)
class ReadState(AppState):

    def register(self):
        self.register_transition('aggregate', Role.COORDINATOR)
        self.register_transition('await', Role.PARTICIPANT)
            
    def run(self):
        c.read_data('/mnt/input/anno.csv', '/mnt/input/exp.csv')

        # find columns to drop

        # to_drop = c.get_dataframe().columns[(c.get_dataframe() == 0).all()]
        to_drop = ['msp_0001']

        self.send_data_to_coordinator(to_drop)

        if self.is_coordinator:
            return 'aggregate'
        else:
            return 'await'

@app_state('aggregate', Role.COORDINATOR)
class AggregateState(AppState):

    def register(self):
        self.register_transition('await', Role.COORDINATOR)

    def run(self):
        to_drop = self.gather_data()

        # find common columns to drop and broadcast them

        self.broadcast_data(to_drop)

        return 'await' 

@app_state('await', Role.BOTH)
class AwaitState(AppState):

    def register(self):
        self.register_transition('send_global_params', Role.COORDINATOR)
        self.register_transition('apply', Role.PARTICIPANT)

    def run(self):
        to_drop = self.await_data()

        # # remove columns from to_drop
        # c.delete_columns(to_drop)

        # apply model here
        model_params = logistic_regression(c.get_dataframe())

        self.send_data_to_coordinator(model_params)

        if self.is_coordinator:
            return 'send_global_params'
        else:
            return 'apply'

@app_state('send_global_params', Role.COORDINATOR)
class SendGlobalParamsState(AppState):

    def register(self):
        self.register_transition('apply', Role.COORDINATOR)

    def run(self):
        final_params = self.gather_data()

        self.broadcast_data(final_params)

        return 'apply'


@app_state('apply', Role.BOTH)
class ApplyState(AppState):

    def register(self):
        self.register_transition('terminal')

    def run(self):
        final_params = self.await_data()

        self.log(final_params)

        return 'terminal'